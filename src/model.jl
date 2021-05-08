struct ResNetModel{E, C, H}
    entry::E
    encoder::C
    head::H
end
Flux.@functor ResNetModel

function (model::ResNetModel)(x::T) where T
    x |> model.entry |> model.encoder |> model.head
end

function make_connection(channels::Pair{Int64, Int64}, stride::Int)
    stride == 1 && channels[1] == channels[2] && return +
    Shortcut(Chain(
        Conv((1, 1), channels; stride, bias=false), BatchNorm(channels[2]),
    ))
end

function make_layer(
    block, channels::Pair{Int64, Int64}, repeat::Int64,
    expansion::Int64, stride::Int64 = 1,
)
    layer = []
    expanded_channels = channels[2] * expansion

    connection = make_connection(channels[1]=>expanded_channels, stride)
    push!(layer, block(channels, stride, connection))

    for i in 2:repeat
        push!(layer, block(expanded_channels=>channels[2]))
    end
    Chain(layer...)
end

function ResNetModel(size::Int64 = 18, classes::Int64 = 1000)
    channels = [64, 128, 256, 512]
    strides = [1, 2, 2, 2]
    repeats, block, expansion = Dict(
        18 => ([2, 2, 2, 2], BasicBlock, 1),
        34 => ([3, 4, 6, 3], BasicBlock, 1),
        50 => ([3, 4, 6, 3], Bottleneck, 4),
        101 => ([3, 4, 23, 3], Bottleneck, 4),
        152 => ([3, 8, 36, 3], Bottleneck, 4),
    )[size]

    entry = Chain(
        Conv((7, 7), 3=>64, pad=3, stride=2, bias=false),
        BatchNorm(64, relu),
        MaxPool((3, 3), pad=1, stride=2),
    )
    head = Chain(MeanPool((7, 7)), flatten, Dense(512 * expansion, classes))

    encoder = []
    in_channels = 64
    for (out_channels, repeat, stride) in zip(channels, repeats, strides)
        push!(encoder, make_layer(
            block, in_channels=>out_channels, repeat, expansion, stride
        ))
        in_channels = out_channels * expansion
    end
    ResNetModel(entry, Chain(encoder...), head)
end
