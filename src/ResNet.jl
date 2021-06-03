module ResNet
export ResNetModel, stages_channels

using BSON
using Pickle
using Downloads: download
using Images

using CUDA
CUDA.allowscalar(false)
using Flux

include("blocks.jl")
include("model.jl")
include("load_utils.jl")

# function main()
#     device = cpu

#     # model = from_pretrained(;model_size=18)
#     model = ResNetModel(;size=18, classes=10)
#     model = model |> device
#     θ = model |> params
#     @info model.size
#     @info stages_channels(model)

#     x = randn(Float32, 224, 224, 3, 1) |> device
#     # y = randn(Float32, 10, 2) |> device

#     features = model(x, Val(:stages))
#     for f in features
#         @info size(f), typeof(f)
#     end

#     o = model(x)
#     @info typeof(o), size(o)

#     # g = gradient(θ) do
#     #     o = x |> model
#     #     Flux.mse(o, y)
#     # end
#     # @info g
# end

# function main()
#     device = gpu
#     model = from_pretrained(;model_size=18)
#     model = model |> testmode! |> device
#     @info "Model loaded."

#     images = [
#         raw"C:\Users\tonys\Downloads\elephant2-r.jpg",
#         raw"C:\Users\tonys\Downloads\spaceshuttle-r.jpg",
#     ]
#     μ = reshape([0.485, 0.456, 0.406], (3, 1, 1))
#     σ = reshape([0.229, 0.224, 0.225], (3, 1, 1))
#     for image in images
#         x = Images.load(image) |> channelview .|> Float32
#         @. x = (x - μ) / σ
#         x = Flux.unsqueeze(permutedims(x, (3, 2, 1)), 4)

#         @info "Image $image ($(size(x))):"
#         o = x |> device |> model |> softmax |> cpu
#         o = sortperm(o[:, 1])
#         @info "Top 5 classes: $(o[end:-1:end - 5] .- 1)"
#     end
# end
# main()

end
