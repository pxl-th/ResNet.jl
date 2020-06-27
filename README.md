# ResNet

Implementation of [ResNet](https://arxiv.org/abs/1512.03385) in Julia language.

Pretrained on ImageNet weights were ported from PyTorch, tested and confirmed to give identical results with the PyTorch's version.

|Model|Weights|
|:-:|:-:|
|ResNet18|[Download](https://drive.google.com/file/d/1cFC6vUoCC0PsALfDpW6BK3A4mG5bIq4z/view?usp=sharing)|
|ResNet34|[Download](https://drive.google.com/file/d/1UJsLcWtab3lPMg5Vq-8OlDrjg-NT7Mso/view?usp=sharing)|
|ResNet50|[Download](https://drive.google.com/file/d/12E1bTVD818FwfA0RsZ8uP3RrrD6vVHzc/view?usp=sharing)|
|ResNet101|[Download](https://drive.google.com/file/d/10E3AD5pCbEefEFPbG4UAXLhmISMvgnt2/view?usp=sharing)|
|ResNet152|[Download](https://drive.google.com/file/d/1NZ-8d9PrnhAsOSdIZNOYop1iA3bYyDex/view?usp=sharing)|

## How to load

Because currently [Flux.jl](https://github.com/FluxML/Flux.jl) defines only bias β and scale γ as trainable parameters
for BatchNorm, you have to redefine `trainable` function for BatchNorm as follows.

```julia
Flux.trainable(bn::Flux.BatchNorm) = (bn.β, bn.γ, bn.μ, bn.σ²)
```

After that you can load weights using [BSON.jl](https://github.com/JuliaIO/BSON.jl).

```julia
model = resnet(18)
path = "./resnet18-pretrained.bson"
@load path parameters
loadweights!(model, parameters)
```

## Inference

Given image one can perform inference simply by calling model

```julia
image = ...
y = model(image)
```

or if you want to extract features

```julia
features = image |> model.entry |> model.encoder
```

or extract list of features

```julia
entry = image |> model.entry
features = [model.encoder[1](entry)]
for encoder in model.encoder[2:end]
    push!(features, encoder(features[end]))
end
```
