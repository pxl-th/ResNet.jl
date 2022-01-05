struct Shortcut{S}
    s::S
end
Flux.@functor Shortcut
(s::Shortcut)(mx, x) = mx + s.s(x)

struct ResidualBlock{B}
    block::B
end
Flux.@functor ResidualBlock
(b::ResidualBlock)(x) = relu.(b.block(x))

function BasicBlock(channels, connection; stride = 1)
    layer = Chain(
        Conv((3, 3), channels; stride, pad=1, bias=false),
        BatchNorm(channels[2], relu),
        Conv((3, 3), channels[2]=>channels[2]; pad=1, bias=false),
        BatchNorm(channels[2]))
    ResidualBlock(SkipConnection(layer, connection))
end

function Bottleneck(channels, connection; stride = 1, expansion = 4)
    layer = Chain(
        Conv((1, 1), channels, bias=false),
        BatchNorm(channels[2], relu),
        Conv((3, 3), channels[2]=>channels[2]; stride, pad=1, bias=false),
        BatchNorm(channels[2], relu),
        Conv((1, 1), channels[2]=>(channels[2] * expansion); bias=false),
        BatchNorm(channels[2] * expansion))
    ResidualBlock(SkipConnection(layer, connection))
end

function make_layer(block, channels, repeat, expansion, stride = 1)
    layer = ResidualBlock[]

    if stride == 1 && channels[1] == channels[2]
        push!(layer, block(channels, +; stride))
    else
        c = Shortcut(Chain(
            Conv((1, 1), channels; stride, bias=false),
            BatchNorm(channels[2])))
        push!(layer, block(channels, c; stride))
    end

    expanded_channels = channels[2] * expansion
    for _ in 2:repeat
        push!(layer, block(expanded_channels=>channels[2], +))
    end
    Chain(layer...)
end
