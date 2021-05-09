module ResNet
export ResNetModel

using Flux

include("blocks.jl")
include("model.jl")

# function main()
#     model = ResNetModel(;size=50, in_channels=1, classes=nothing)
#     x = randn(Float32, 224, 224, 1, 1)
#     features = model(x, Val(:stages))
#     for f in features
#         @info size(f)
#     end
#     @info model.stages_channels
# end
# main()

end
