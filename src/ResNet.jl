module ResNet
export ResNetModel

using Flux

include("blocks.jl")
include("model.jl")

# function main()
#     model = ResNetModel(18, 10)
#     x = randn(Float32, 224, 224, 3, 1)
#     o = x |> model
#     @info size(o)

#     features = []
#     o = x |> model.entry
#     for enc in model.encoder
#         o = o |> enc
#         @info size(o)
#         push!(features, o)
#     end
#     @info length(features)
# end
# main()

end
