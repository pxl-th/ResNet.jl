module ResNet
export ResNetModel, stages_channels

using Pickle
using Downloads: download
using Flux

include("blocks.jl")
include("model.jl")
include("load_utils.jl")

# using Images
# function main()
#     device = cpu
#     model = from_pretrained(18)
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
