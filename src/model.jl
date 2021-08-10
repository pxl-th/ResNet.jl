struct ResNetModel
    entry
    pooling
    layers
    head

    size::Int64
end
Flux.@functor ResNetModel

(m::ResNetModel)(x) = x |> m.entry |> m.pooling |> m.layers |> m.head

function (m::ResNetModel)(x, ::Val{:stages})
    stages = typeof(x)[]

    o = x |> m.entry
    push!(stages, o)
    o = o |> m.pooling

    for l in m.layers
        o = o |> l
        push!(stages, o)
    end
    stages
end

const Config = Dict(
    18=>((2, 2, 2, 2), BasicBlock, 1),
    34=>((3, 4, 6, 3), BasicBlock, 1),
    50=>((3, 4, 6, 3), Bottleneck, 4),
    101=>((3, 4, 23, 3), Bottleneck, 4),
    152=>((3, 8, 36, 3), Bottleneck, 4),
)

function ResNetModel(
    model_size::Int64 = 18;
    in_channels::Int64 = 3,
    classes::Union{Int64, Nothing} = 1000,
)
    if !(model_size in keys(Config))
        throw(
            "Invalid mode size [$model_size]. " *
            "Supported sizes are $(keys(Config))."
        )
    end
    repeats, block, expansion = Config[model_size]

    entry = Chain(
        Conv((7, 7), in_channels=>64, pad=(3, 3), stride=(2, 2), bias=false),
        BatchNorm(64, relu),
    )
    pooling = MaxPool((3, 3), pad=(1, 1), stride=(2, 2))

    head = nothing
    if classes ≢ nothing
        head = Chain(
            MeanPool((7, 7)), flatten, Dense(512 * expansion, classes),
        )
    end

    in_channels = 64
    channels = (64, 128, 256, 512)
    strides = (1, 2, 2, 2)

    layers = []
    for (out_channels, repeat, stride) in zip(channels, repeats, strides)
        push!(layers, make_layer(
            block, in_channels=>out_channels, repeat, expansion, stride,
        ))
        in_channels = out_channels * expansion
    end

    ResNetModel(entry, pooling, Chain(layers...), head, model_size)
end

function stages_channels(r::ResNetModel)
    e = Config[r.size][3]
    (64, 64 * e, 128 * e, 256 * e, 512 * e)
end
