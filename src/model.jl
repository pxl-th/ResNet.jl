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

"""
```julia
resnet(size::Int = 18, classes::Int = 1000)
```

Construct ResNet model.

# Parameters
- `size::Int = 18`: Size of the ResNet model.
  Available sizes are `18, 34, 50, 101, 152`.
- `classes::Int = 1000`: Number of classes in the last dense layer.
"""
function resnet(size::Int = 18, classes::Int = 1000)::ResNetModel
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
        layer = make_layer(in_filters, out_filters, repeat, block, expansion, stride)
        push!(encoder, layer)
        in_filters = out_filters * expansion
    end
    ResNetModel(entry, Chain(encoder...), head)
end
