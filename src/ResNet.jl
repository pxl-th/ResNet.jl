module ResNet
export ResNetModel, stages_channels

using BSON
using Downloads: download

using CUDA
CUDA.allowscalar(false)
using Flux

include("blocks.jl")
include("model.jl")
# include("load_utils.jl")

# function main()
#     device = cpu

#     model = ResNetModel(;size=18, in_channels=3, classes=4, use_bn=true)
#     model = model |> device
#     @info model.size
#     @info stages_channels(model)

#     θ = model |> params
#     for t in θ
#         @info typeof(t)
#     end

#     x = randn(Float32, 224, 224, 3, 2) |> device
#     y = randn(Float32, 4, 2) |> device

#     features = model(x, Val(:stages))
#     for f in features
#         @info size(f), typeof(f)
#     end

#     o = model(x)
#     @info typeof(o), size(o)

#     g = gradient(θ) do
#         o = x |> model
#         Flux.mse(o, y)
#     end
#     @info g
# end
# main()

end
