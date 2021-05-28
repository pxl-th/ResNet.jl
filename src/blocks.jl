function maybe_bn(channels::Int64, use_bn::Bool)
    use_bn && return BatchNorm(channels)
    identity
end

function BasicBlock(
    channels::Pair{Int64, Int64}, stride::Int64 = 1, connection = +;
    use_bn::Bool,
)
    activation = x -> x .|> relu
    layer = Chain(
        Conv((3, 3), channels; stride, pad=1, bias=false),
        maybe_bn(channels[2], use_bn),
        activation,
        Conv((3, 3), channels[2]=>channels[2]; pad=1, bias=false),
        maybe_bn(channels[2], use_bn),
    )
    Chain(SkipConnection(layer, connection), x -> relu.(x))
end

function Bottleneck(
    channels::Pair{Int64, Int64}, stride::Int = 1, connection = +,
    expansion::Int = 4; use_bn::Bool,
)
    activation = x -> x .|> relu
    layer = Chain(
        Conv((1, 1), channels, bias=false),
        maybe_bn(channels[2], use_bn),
        activation,
        Conv((3, 3), channels[2]=>channels[2]; stride, pad=1, bias=false),
        maybe_bn(channels[2], use_bn),
        activation,
        Conv((1, 1), channels[2]=>(channels[2] * expansion); bias=false),
        maybe_bn(channels[2] * expansion, use_bn),
    )
    Chain(SkipConnection(layer, connection), activation)
end

struct Shortcut{S}
    s::S
end
Flux.@functor Shortcut

function (s::Shortcut)(mx::AbstractArray{T}, x::AbstractArray{T}) where T
    mx + s.s(x)
end

function make_connection(channels::Pair{Int64, Int64}, stride::Int64)
    stride == 1 && channels[1] == channels[2] && return +
    Shortcut(Chain(
        Conv((1, 1), channels; stride, bias=false),
        BatchNorm(channels[2]),
    ))
end

function make_layer(
    block, channels::Pair{Int64, Int64}, repeat::Int64,
    expansion::Int64, stride::Int64 = 1; use_bn::Bool,
)
    expanded_channels = channels[2] * expansion
    connection = make_connection(channels[1]=>expanded_channels, stride)

    layer = []
    push!(layer, block(channels, stride, connection; use_bn))
    for i in 2:repeat
        push!(layer, block(expanded_channels=>channels[2]; use_bn))
    end
    Chain(layer...)
end
