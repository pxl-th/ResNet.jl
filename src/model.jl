struct ResNetModel{E, P, C, H}
    entry::E
    pooling::P
    layers::C
    head::H

    stages_channels::Vector{Int64}
end
Flux.@functor ResNetModel

function ResNetModel(;
    size::Int64 = 18,
    in_channels::Int64 = 3,
    classes::Union{Int64, Nothing} = 1000,
)
    config = Dict(
        18=>([2, 2, 2, 2], BasicBlock, 1),
        34=>([3, 4, 6, 3], BasicBlock, 1),
        50=>([3, 4, 6, 3], Bottleneck, 4),
        101=>([3, 4, 23, 3], Bottleneck, 4),
        152=>([3, 8, 36, 3], Bottleneck, 4),
    )
    if !(size in keys(config))
        throw(
            "Invalid size if the model [$size]. " *
            "Supported sizes are $(keys(config))."
        )
    end

    channels = [64, 128, 256, 512]
    strides = [1, 2, 2, 2]
    repeats, block, expansion = config[size]
    stages_channels = [
        in_channels, 64, 64 * expansion,
        128 * expansion, 256 * expansion, 512 * expansion,
    ]

    entry = Chain(
        Conv((7, 7), in_channels=>64, pad=3, stride=2, bias=false),
        BatchNorm(64, relu),
    )
    pooling = MaxPool((3, 3), pad=1, stride=2)

    if classes ≢ nothing
        head = Chain(
            MeanPool((7, 7)), flatten, Dense(512 * expansion, classes),
        )
    else
        head = nothing
    end

    layers = []
    in_channels = 64
    for (out_channels, repeat, stride) in zip(channels, repeats, strides)
        push!(layers, make_layer(
            block, in_channels=>out_channels, repeat, expansion, stride
        ))
        in_channels = out_channels * expansion
    end

    ResNetModel(entry, pooling, Chain(layers...), head, stages_channels)
end

function (m::ResNetModel)(x::AbstractArray{T}) where T
    x |> m.entry |> m.pooling |> m.layers |> m.head
end

function (m::ResNetModel)(x, ::Val{:stages})
    stages = Vector{typeof(x)}(undef, 0)
    push!(stages, x)

    o = x |> m.entry
    push!(stages, o)
    o = o |> m.pooling

    for l in m.layers
        o = o |> l
        push!(stages, o)
    end

    stages
end
