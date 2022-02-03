using Test
# using Flux
using ResNet
import BenchmarkTools: @btime
const Flux = ResNet.Flux

@testset "Test ResNet forward/backward passes" begin
    in_channels = 3
    classes = 10
    N = 5

    host_model = ResidualNetwork(18; in_channels, classes)
    host_x = randn(Float32, 224, 224, in_channels, N)
    host_y = randn(Float32, classes, N)

    for transfer in (Flux.cpu ∘ Flux.f32, Flux.gpu ∘ Flux.f32)
        @show transfer

        model = Flux.trainmode!(transfer(host_model))
        θ = Flux.params(model)
        x = transfer(host_x)
        y = transfer(host_y)

        @info "forward"
        @time model(x)
        @btime $model($x)

        @inferred model(x)
        @inferred Flux.crossentropy(Flux.softmax(model(x)), y)

        @info "backward"
        @time Flux.gradient(θ) do
            Flux.crossentropy(Flux.softmax(model(x)), y)
        end
        @btime Flux.gradient($θ) do
            Flux.crossentropy(Flux.softmax($model($x)), $y)
        end

        endpoints = model(x, Val(:stages))
        @test length(endpoints) == 5
    end
end
