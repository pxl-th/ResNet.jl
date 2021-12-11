# ResNet

Create model:

```julia
model = ResNetModel(18; in_channels=3, classes=10)
```

or you can ommit classification layer,
by specifying number of `classes` as `nothing`.

- simple:

```julia
y = x |> model
```

- extract list of features:

```julia
features = model(x, Val(:stages))
```

Number of channels for each element of `features` is in `model.stages`.
Size of the model is in `model.size`.
