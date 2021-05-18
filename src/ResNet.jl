module ResNet
export ResNetModel, stages_channels

using CUDA
CUDA.allowscalar(false)
using Flux

include("blocks.jl")
include("model.jl")

end
