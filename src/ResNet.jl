module ResNet
export ResNetModel, stages_channels

using CUDA
CUDA.allowscalar(false)
using Flux

include("blocks.jl")
include("model.jl")

# function main()
#     T = Float16
#     device = gpu
#     @info "Type $T"

#     model = ResNetModel(;size=18, in_channels=3, classes=4, use_bn=true)
#     model = Flux.paramtype(T, model)
#     model = model |> device
#     @info model.size
#     @info stages_channels(model)

#     θ = model |> params
#     for t in θ
#         @info typeof(t)
#     end

#     x = randn(T, 224, 224, 3, 2) |> device
#     y = randn(T, 4, 2) |> device

#     # features = model(x, Val(:stages))
#     # for f in features
#     #     @info size(f), typeof(f)
#     # end

#     o = model(x)
#     @info typeof(o), size(o)

#     g = gradient(θ) do
#         o = x |> model
#         Flux.mse(o, y)
#     end
#     @info g
# end
# main()

# function test_block()
#     T = Float16
#     device = gpu

#     block = BasicBlock(3=>3; use_bn=true)
#     # block = BatchNorm(3; ϵ=Float16(1f-5), momentum=Float16(0.1f0))
#     block = Flux.paramtype(T, block)
#     block = block |> device
#     @info typeof(block)

#     x = randn(T, 10, 10, 3, 2) |> device
#     o = x |> block
#     @info typeof(o), size(o)
# end
# test_block()

end
