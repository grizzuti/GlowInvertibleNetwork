using GlowInvertibleNetwork, InvertibleNetworks, LinearAlgebra, Test, Flux, Random
include("./test_utils.jl")
Random.seed!(42)

# Dimensions
n = 16
nc = 4
batchsize = 3
clamp = 3

step = 1e-6
rtol = 1e-5

device = cpu
# device = gpu

for N = 1:3

    # Test invertibility
    S = ExpClampLayerNew(; clamp=clamp)
    X = randn(Float32, n*ones(Int, N)..., nc, batchsize) |> device
    Y = randn(Float32, n*ones(Int, N)..., nc, batchsize) |> device
    @test X ≈ S.inverse(S.forward(X)) rtol=1f-6

    # Gradient test (input)
    S = ExpClampLayerNew(; clamp=clamp)
    X  = randn(Float64, n*ones(Int, N)..., nc, batchsize) |> device; X = Float64.(X)
    gradient_test_input(S, X; step=step, rtol=rtol, invnet=false)

end