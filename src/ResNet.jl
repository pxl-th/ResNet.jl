module ResNet
export ResNetModel, stages_channels

using Flux

include("blocks.jl")
include("model.jl")

# function main()
#     model = ResNetModel(;size=18, in_channels=3, classes=10, use_bn=false) |> gpu
#     @info model.size
#     @info stages_channels(model)

#     x = randn(Float32, 224, 224, 3, 1) |> gpu
#     o = x |> model
#     @info size(o)

#     features = model(x, Val(:stages))
#     for f in features
#         @info size(f), typeof(f)
#     end
# end
# main()

end
