using Test
using Flux
using ResNet

@testset "Test ResNet forward/backward passes" begin
    device = gpu

    N, in_channels = 3, 5
    model = ResidualNetwork(18; in_channels, classes=10)
    model = model |> trainmode! |> device
    trainables = model |> params

    x = randn(Float32, 224, 224, in_channels, N) |> device
    y = randn(Float32, 10, N) |> device

    x |> model
    endpoints = model(x, Val(:stages))
    @show size.(endpoints)
    @test length(endpoints) == 5

    @time Flux.crossentropy(softmax(model(x)), y)
    @time gradient(trainables) do
        Flux.crossentropy(softmax(model(x)), y)
    end
end
