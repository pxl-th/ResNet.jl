Flux.trainable(bn::Flux.BatchNorm) = (bn.β, bn.γ, bn.μ, bn.σ²)

struct Shortcut
    shortcut
end
Flux.@functor Shortcut

function (shortcut::Shortcut)(mx::T, x::T) where T
    mx + shortcut.shortcut(x)
end

struct ResNetModel
    entry::Chain
    encoder::Chain
    head::Chain
end
Flux.@functor ResNetModel

function (model::ResNetModel)(x::T) where T
    x |> model.entry |> model.encoder |> model.head
end

function BasicBlock(
    in_filters::Int, out_filters::Int, stride::Int = 1, connection = +,
)::Chain
    layer = Chain(
        Conv((3, 3), in_filters => out_filters, stride=stride, pad=1, bias=Flux.Zeros(Float32)),
        BatchNorm(out_filters, relu),
        Conv((3, 3), out_filters => out_filters, pad=1, bias=Flux.Zeros(Float32)),
        BatchNorm(out_filters),
    )
    Chain(SkipConnection(layer, connection), x -> relu.(x))
end

function Bottleneck(
    in_filters::Int, out_filters::Int, stride::Int = 1, connection = +,
    expansion::Int = 4,
)::Chain
    layer = Chain(
        Conv((1, 1), in_filters => out_filters, bias=Flux.Zeros(Float32)),
        BatchNorm(out_filters, relu),
        Conv((3, 3), out_filters => out_filters, stride=stride, pad=1, bias=Flux.Zeros(Float32)),
        BatchNorm(out_filters, relu),
        Conv((1, 1), out_filters => out_filters * expansion, bias=Flux.Zeros(Float32)),
        BatchNorm(out_filters * expansion),
    )
    Chain(SkipConnection(layer, connection), x -> relu.(x))
end

function make_connection(in_filters::Int, out_filters::Int, stride::Int)
    if stride == 1 && in_filters == out_filters
        return +
    end
    Shortcut(Chain(
        Conv((1, 1), in_filters => out_filters, stride=stride, bias=Flux.Zeros(Float32)),
        BatchNorm(out_filters)
    ))
end

function make_layer(
    in_filters::Int, out_filters::Int, repeat::Int,
    block, expansion::Int, stride::Int = 1,
)::Chain
    connection = make_connection(in_filters, out_filters * expansion, stride)
    layer = [block(in_filters, out_filters, stride, connection)]
    in_filters = out_filters * expansion
    for i in 2:repeat
        push!(layer, block(in_filters, out_filters))
    end
    Chain(layer...)
end

function resnet(size::Int = 18, classes::Int = 1000)
    filters = [64, 128, 256, 512]
    strides = [1, 2, 2, 2]
    repeats, block, expansion = Dict(
        18 => ([2, 2, 2, 2], BasicBlock, 1),
        34 => ([3, 4, 6, 3], BasicBlock, 1),
        50 => ([3, 4, 6, 3], Bottleneck, 4),
        101 => ([3, 4, 23, 3], Bottleneck, 4),
        152 => ([3, 8, 36, 3], Bottleneck, 4),
    )[size]

    entry = Chain(
        Conv((7, 7), 3 => 64, pad=(3, 3), stride=(2, 2), bias=Flux.Zeros(Float32)),
        BatchNorm(64, relu),
        MaxPool((3, 3), pad=(1, 1), stride=(2, 2)),
    )
    head = Chain(MeanPool((7, 7)), flatten, Dense(512 * expansion, classes))

    encoder = []
    in_filters = 64
    for (out_filters, repeat, stride) in zip(filters, repeats, strides)
        layer = make_layer(
            in_filters, out_filters, repeat,
            block, expansion, stride,
        )
        push!(encoder, layer)
        in_filters = out_filters * expansion
    end
    ResNetModel(entry, Chain(encoder...), head)
end

function rebuild_conv(x, shape)
    w = Array{Float32}(undef, shape...)
    filter_x, filter_y = shape[1:2] .+ 1
    @inbounds for (i, j, k, m) in Iterators.product([1:s for (i, s) in enumerate(shape)]...)
        w[filter_x - i, filter_y - j, k, m] = Float32(x[m][k][j][i])
    end
    w
end

function rebuild_dense(x, shape)
    w = Array{Float32}(undef, shape...)
    @inbounds for (i, j) in Iterators.product([1:s for s in shape]...)
        w[i, j] = Float32(x[i][j])
    end
    w
end

function loadweights!(model, weights)
    for (p, w) in zip(params(model), weights)
        if length(p) == 1
            continue
        end
        copyto!(p, w)
    end
end

function load_conv(conv, paths, pid)
    path = paths[pid]
    @load path p
    copyto!(conv.weight, rebuild_conv(p, size(conv.weight)))
    pid += 1
    pid
end

function load_bn(bn, paths, pid)
    path = paths[pid]
    @load path p
    copyto!(bn.γ, p)
    pid += 1
    path = paths[pid]
    @load path p
    copyto!(bn.β, p)
    pid += 1
    path = paths[pid]
    @load path p
    copyto!(bn.μ, p)
    pid += 1
    path = paths[pid]
    @load path p
    copyto!(bn.σ², p)
    pid += 1
    pid
end

function load_dense(d, paths, pid)
    path = paths[pid]
    @load path p
    copyto!(d.W, rebuild_dense(p, size(d.W)))
    pid += 1
    path = paths[pid]
    @load path p
    copyto!(d.b, p)
    pid += 1
    pid
end

function load_entry(entry, paths, pid)
    if entry isa Conv
        pid = load_conv(entry, paths, pid)
    elseif entry isa BatchNorm
        pid = load_bn(entry, paths, pid)
    elseif entry isa Dense
        pid = load_dense(entry, paths, pid)
    elseif entry isa Chain
        for e in entry
            pid = load_entry(e, paths, pid)
        end
    elseif entry isa SkipConnection
        pid = load_entry(entry.layers, paths, pid)
        pid = load_entry(entry.connection, paths, pid)
    elseif entry isa Shortcut
        pid = load_entry(entry.shortcut, paths, pid)
    end
    pid
end


function load()
    path = "resnet101-pretrained.bson"
    dir = raw"C:\Users\tonys\projects\julia\ResNet\resnet101-pretrained"
    weights_paths = readdir(dir, join=true)

    pid = 1
    model = resnet(101, 1000)
    @info "Loading model"
    pid = load_entry(model.entry, weights_paths, pid)
    pid = load_entry(model.encoder, weights_paths, pid)
    pid = load_entry(model.head, weights_paths, pid)
    @info "Loaded"

    parameters = params(model) .|> cpu
    @save path parameters
    # @load path parameters
    # loadweights!(model, parameters)
    testmode!(model)

    images = [
        raw"C:\Users\tonys\Downloads\kazan.jpg",
        raw"C:\Users\tonys\Downloads\el.jpg",
        raw"C:\Users\tonys\Downloads\pug.jpg",
    ]
    for path in images
        image = FileIO.load(path)
        image = Float32.(Images.channelview(image))
        image = permutedims(image, (3, 2, 1))
        image = Flux.unsqueeze(image, 4)

        y = Flux.softmax(model(image)[:, 1])
        top5 = sortperm(y)[end - 4:end]
        println(path)
        println(top5 .- 1)
        println(y[top5])
        println("===========")
    end
    nothing
end
