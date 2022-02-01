module ResNet
export ResidualNetwork

using Flux

include("blocks.jl")
include("model.jl")

# function main()
#     device = gpu
#     precision = f32
#     in_channels = 1
#     N = 1

#     transfer = device ∘ precision
#     model = trainmode!(transfer(ResidualNetwork(18; in_channels, classes=10)))
#     θ = params(model)
#     x = transfer(randn(Float32, 224, 224, in_channels, N))
#     y = transfer(randn(Float32, 10, N))

#     @time model(x)
#     Flux.crossentropy(softmax(model(x)), y)
#     @time gradient(θ) do
#         Flux.crossentropy(softmax(model(x)), y)
#     end
# end
# main()

end
