struct ResidualNetwork{E, P, L, H}
    entry::E
    pooling::P
    layers::L
    head::H

    size::Int64
    stages::NTuple{5, Int64}
end
Flux.@functor ResidualNetwork

(m::ResidualNetwork)(x) = m.head(m.layers(m.pooling(m.entry(x))))
(m::ResidualNetwork)(x, ::Val{:stages}) = Flux.extraChain((
    m.entry, Chain(m.pooling, m.layers[1]),
    m.layers[2], m.layers[3], m.layers[4]), x)

function ResidualNetwork(model_size = 18; in_channels = 3, classes = 1000)
    config = Dict(
        18=>((2, 2, 2, 2), BasicBlock, 1),
        34=>((3, 4, 6, 3), BasicBlock, 1),
        50=>((3, 4, 6, 3), Bottleneck, 4),
        101=>((3, 4, 23, 3), Bottleneck, 4),
        152=>((3, 8, 36, 3), Bottleneck, 4))

    model_size in keys(config) || throw(
        "Invalid mode size [$model_size]. Supported sizes are $(keys(config)).")

    repeats, block, expansion = config[model_size]
    stages_channels = _get_stages_channels(expansion)

    entry = Chain(
        Conv((7, 7), in_channels=>64, pad=(3, 3), stride=(2, 2), bias=false),
        BatchNorm(64, relu))
    pooling = MaxPool((3, 3), pad=(1, 1), stride=(2, 2))

    head = nothing
    if classes ≢ nothing
        head = Chain(MeanPool((7, 7)), Flux.flatten, Dense(512 * expansion, classes))
    end

    in_channels = 64
    channels = (64, 128, 256, 512)
    strides = (1, 2, 2, 2)

    layers = []
    for (out_channels, repeat, stride) in zip(channels, repeats, strides)
        push!(layers, make_layer(
            block, in_channels=>out_channels, repeat, expansion, stride))
        in_channels = out_channels * expansion
    end

    ResidualNetwork(
        entry, pooling, Chain(layers...), head,
        model_size, stages_channels)
end

@inline _get_stages_channels(expansion) = (
    64, 64 * expansion, 128 * expansion, 256 * expansion, 512 * expansion)
