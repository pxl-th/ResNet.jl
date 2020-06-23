module ResNet
export resnet

using BSON: @save, @load
using FileIO
using Images
using Statistics

using CuArrays
using Flux
using Flux.Optimise: ADAM, update!
using Flux.Data: DataLoader

include("model.jl")

end
