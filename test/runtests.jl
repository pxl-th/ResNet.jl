using Test
using Flux
using ResNet

@testset "Test ResNet forward/backward passes" begin
    in_channels = 3
    classes = 10
    N = 5

    host_model = ResidualNetwork(18; in_channels, classes)
    host_x = randn(Float32, 224, 224, in_channels, N)
    host_y = randn(Float32, classes, N)

    for transfer in (cpu ∘ f32, gpu ∘ f32)
        @show transfer

        model = trainmode!(transfer(host_model))
        θ = params(model)
        x = transfer(host_x)
        y = transfer(host_y)

        println("Forward timing:")
        @time model(x)
        @time model(x)

        @inferred model(x)
        @inferred Flux.crossentropy(softmax(model(x)), y)

        println("Backward timing:")
        @time gradient(θ) do
            Flux.crossentropy(softmax(model(x)), y)
        end
        @time gradient(θ) do
            Flux.crossentropy(softmax(model(x)), y)
        end

        endpoints = model(x, Val(:stages))
        @test length(endpoints) == 5
    end
end
