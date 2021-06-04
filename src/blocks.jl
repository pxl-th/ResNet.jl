struct Shortcut
    s
end
Flux.@functor Shortcut
(s::Shortcut)(mx, x) = mx + s.s(x)

struct ResidualBlock
    block
end
Flux.@functor ResidualBlock
(b::ResidualBlock)(x) = x |> b.block .|> relu

function BasicBlock(
    channels::Pair{Int64, Int64}, connection; stride::Int64 = 1,
)
    layer = Chain(
        Conv((3, 3), channels; stride, pad=1, bias=false),
        BatchNorm(channels[2], relu),
        Conv((3, 3), channels[2]=>channels[2]; pad=1, bias=false),
        BatchNorm(channels[2]),
    )
    ResidualBlock(SkipConnection(layer, connection))
end

function Bottleneck(
    channels::Pair{Int64, Int64}, connection;
    stride::Int = 1, expansion::Int = 4,
)
    layer = Chain(
        Conv((1, 1), channels, bias=false),
        BatchNorm(channels[2], relu),
        Conv((3, 3), channels[2]=>channels[2]; stride, pad=1, bias=false),
        BatchNorm(channels[2], relu),
        Conv((1, 1), channels[2]=>(channels[2] * expansion); bias=false),
        BatchNorm(channels[2] * expansion),
    )
    ResidualBlock(SkipConnection(layer, connection))
end

function make_layer(
    block, channels::Pair{Int64, Int64}, repeat::Int64,
    expansion::Int64, stride::Int64 = 1,
)
    expanded_channels = channels[2] * expansion
    if stride == 1 && channels[1] == channels[2]
        connection = +
    else
        connection = Shortcut(Chain(
            Conv((1, 1), channels; stride, bias=false),
            BatchNorm(channels[2]),
        ))
    end

    layer = ResidualBlock[]
    push!(layer, block(channels, connection; stride))
    for i in 2:repeat
        push!(layer, block(expanded_channels=>channels[2], +))
    end
    Chain(layer...)
end
