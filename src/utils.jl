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
    path = "resnet18-pretrained.bson"
    dir = raw"C:\Users\tonys\projects\julia\ResNet\resnet18-pretrained"
    weights_paths = readdir(dir, join=true)

    pid = 1
    model = resnet(18, 1000)
    # @info "Loading model"
    # pid = load_entry(model.entry, weights_paths, pid)
    # pid = load_entry(model.encoder, weights_paths, pid)
    # pid = load_entry(model.head, weights_paths, pid)
    # @info "Loaded"

    # parameters = params(model) .|> cpu
    # @save path parameters
    @load path parameters
    loadweights!(model, parameters)
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

        entry = image |> model.entry
        features = [model.encoder[1](entry)]
        for e in model.encoder[2:end]
            push!(features, e(features[end]))
        end
        y = Flux.softmax(model.head(features[end])[:, 1])

        # y = Flux.softmax(model(image)[:, 1])
        top5 = sortperm(y)[end - 4:end]
        println(path)
        println(top5 .- 1)
        println(y[top5])
        println("===========")
    end
    nothing
end
