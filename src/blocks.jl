function BasicBlock(
    channels::Pair{Int64, Int64}, stride::Int64 = 1, connection = +,
)
    layer = Chain(
        Conv((3, 3), channels; stride, pad=1, bias=false),
        BatchNorm(channels[2], relu),
        Conv((3, 3), channels[2]=>channels[2]; pad=1, bias=false),
        BatchNorm(channels[2]),
    )
    Chain(SkipConnection(layer, connection), x -> relu.(x))
end

function Bottleneck(
    channels::Pair{Int64, Int64}, stride::Int = 1, connection = +,
    expansion::Int = 4,
)
    layer = Chain(
        Conv((1, 1), channels, bias=false),
        BatchNorm(channels[2], relu),
        Conv((3, 3), channels[2]=>channels[2]; stride, pad=1, bias=false),
        BatchNorm(channels[2], relu),
        Conv((1, 1), channels[2]=>(channels[2] * expansion); bias=false),
        BatchNorm(channels[2] * expansion),
    )
    Chain(SkipConnection(layer, connection), x -> relu.(x))
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
        Conv((1, 1), channels; stride, bias=false), BatchNorm(channels[2]),
    ))
end

function make_layer(
    block, channels::Pair{Int64, Int64}, repeat::Int64,
    expansion::Int64, stride::Int64 = 1,
)
    expanded_channels = channels[2] * expansion
    connection = make_connection(channels[1]=>expanded_channels, stride)

    layer = []
    push!(layer, block(channels, stride, connection))
    for i in 2:repeat
        push!(layer, block(expanded_channels=>channels[2]))
    end
    Chain(layer...)
end
