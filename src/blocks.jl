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
