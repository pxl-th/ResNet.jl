# ResNet

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
features = x |> model.entry |> model.pooling |> model.layers
```

- extract list of features:

```julia
features = model(x, Val(:stages))
```

Number of channels for each element of `features`
can be retrieved by `stages_channels(model)` method.

Size of the model is in `model.size`.
