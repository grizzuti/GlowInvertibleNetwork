using GlowInvertibleNetwork, InvertibleNetworks, LinearAlgebra, Test, Flux, Random
device = InvertibleNetworks.CUDA.functional() ? gpu : cpu
Random.seed!(11)

# Dimensions
n = 64
nc = 4
batchsize = 3

for N = 1:3, do_reverse = [false, true]

    # Test invertibility
    C = OrthogonalConv1x1(nc; logdet=true) |> device; do_reverse && (C = reverse(C))
    X = randn(Float32, n*ones(Int, N)..., nc, batchsize) |> device
    Y = randn(Float32, n*ones(Int, N)..., nc, batchsize) |> device
    @test X ≈ C.inverse(C.forward(X)[1]) rtol=1f-6
    @test Y ≈ C.forward(C.inverse(Y))[1] rtol=1f-6


    # Test backward/inverse coherence
    ΔY = randn(Float32, n*ones(Int, N)..., nc, batchsize) |> device
    Y  = randn(Float32, n*ones(Int, N)..., nc, batchsize) |> device
    X_ = C.inverse(Y)
    _, X = C.backward(ΔY, Y)
    @test X ≈ X_ rtol=1f-6


    # Gradient test (input)
    ΔY = randn(Float32, n*ones(Int, N)..., nc, batchsize) |> device
    ΔX = randn(Float32, n*ones(Int, N)..., nc, batchsize) |> device
    X  = randn(Float32, n*ones(Int, N)..., nc, batchsize) |> device
    Y = C.forward(X)[1]
    ΔX_, _ = C.backward(ΔY, Y)
    @test dot(ΔX, ΔX_) ≈ dot(C.forward(ΔX)[1], ΔY) rtol=1f-4


    # Gradient test (parameters)
    C = OrthogonalConv1x1(nc; logdet=true) |> device; do_reverse && (C = reverse(C))
    C.stencil_pars.data = Float64.(C.stencil_pars.data)
    X  = randn(T, n*ones(Int, N)..., nc, batchsize) |> device; X = Float64.(X)
    ΔY_ = randn(T, n*ones(Int, N)..., nc, batchsize) |> device; ΔY_ = Float64.(ΔY_)
    θ = copy(C.stencil_pars.data)
    Δθ = randn(T, size(θ)) |> device; Δθ = Float64.(Δθ); Δθ *= norm(θ)/norm(Δθ)

    t = Float64(1e-5)
    set_params!(C, [Parameter(θ+t*Δθ/2)])
    Yp1 = C.forward(X)[1]
    set_params!(C, [Parameter(θ-t*Δθ/2)])
    Ym1 = C.forward(X)[1]
    ΔY = (Yp1-Ym1)/t
    set_params!(C, [Parameter(θ)])
    Y = C.forward(X)[1]
    C.backward(ΔY_, Y)
    Δθ_ = C.stencil_pars.grad

    @test dot(ΔY, ΔY_) ≈ dot(Δθ, Δθ_) rtol=Float64(1e-4)

end