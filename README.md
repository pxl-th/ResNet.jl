# ResNet

## Inference

Create model:

```julia
model = ResNetModel(34, 10)
```

- simple:

```julia
y = x |> model
```

- extract features:

```julia
features = x |> model.entry |> model.encoder
```

- extract list of features:

```julia
features = []
o = x |> model.entry
for enc in model.encoder
    o = o |> enc
    push!(features, o)
end
```
