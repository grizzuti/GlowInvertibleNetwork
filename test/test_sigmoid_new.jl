using GlowInvertibleNetwork, InvertibleNetworks, LinearAlgebra, Test, Flux, Random
device = InvertibleNetworks.CUDA.functional() ? gpu : cpu
Random.seed!(11)

# Dimensions
n = 64
nc = 4
batchsize = 3

for N = 1:3

    # Test invertibility
    S = SigmoidLayerNew(; low=0.5f0, high=1.1f0)
    X = randn(Float32, n*ones(Int, N)..., nc, batchsize) |> device
    Y = randn(Float32, n*ones(Int, N)..., nc, batchsize) |> device
    @test X ≈ S.inverse(S.forward(X)) rtol=1f-6

    # Gradient test (input)
    S = SigmoidLayerNew(; low=0.5, high=1.1)
    X  = randn(Float64, n*ones(Int, N)..., nc, batchsize) |> device; X = Float64.(X)
    ΔX = randn(Float64, n*ones(Int, N)..., nc, batchsize) |> device; ΔX *= norm(X)/norm(ΔX); ΔX = Float64.(ΔX)
    t = 1e-6
    Yp1 = S.forward(X+t*ΔX/2)
    Ym1 = S.forward(X-t*ΔX/2)
    ΔY_ = (Yp1-Ym1)/t
    ΔY = S.backward(ΔX, X; X=X)

    @test ΔY ≈ ΔY_ rtol=1e-4

end