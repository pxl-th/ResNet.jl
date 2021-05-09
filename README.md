# ResNet

## Inference

Create model:

```julia
model = ResNetModel(;size=34, in_channels=3, classes=10)
```

or you can ommit classification layer,
by specifying number of `classes` as `nothing`.

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
features = model(x, Val(:stages))
```
