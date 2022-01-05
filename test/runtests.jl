using Test
using Flux
using ResNet

@testset "Test ResNet forward/backward passes" begin
    device = gpu
    in_channels = 3
    N = 5

    model = device(trainmode!(ResidualNetwork(18; in_channels, classes=10)))

    θ = params(model)
    x = device(randn(Float32, 224, 224, in_channels, N))
    y = device(randn(Float32, 10, N))

    println("Forward timing:")
    @time Flux.crossentropy(softmax(model(x)), y)

    println("Backward timing:")
    @time gradient(θ) do
        Flux.crossentropy(softmax(model(x)), y)
    end

    endpoints = model(x, Val(:stages))
    @test length(endpoints) == 5
end
