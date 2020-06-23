# ResNet

Implementation of ResNet in Julia language.

Pretrained weights on ImageNet.

Ported from PyTorch, tested and confirmed to give identical results with the PyTorch's version.

|Model|URL|
|:-:|:-:|
|ResNet18|[Download](https://drive.google.com/file/d/1cFC6vUoCC0PsALfDpW6BK3A4mG5bIq4z/view?usp=sharing)|
|ResNet34|[Download](https://drive.google.com/file/d/1UJsLcWtab3lPMg5Vq-8OlDrjg-NT7Mso/view?usp=sharing)|
|ResNet50|Coming soon...|

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

## TODO

- ResNet50
- More docs
