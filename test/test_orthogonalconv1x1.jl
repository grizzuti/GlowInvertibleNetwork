using GlowInvertibleNetwork, InvertibleNetworks, LinearAlgebra, Test, Flux, Random
device = InvertibleNetworks.CUDA.functional() ? gpu : cpu
Random.seed!(11)

# Dimensions
n = 64
nc = 4
batchsize = 3

for N = 1:3

    # Test invertibility
    C = OrthogonalConv1x1(nc; logdet=false) |> device
    X = randn(Float32, n*ones(Int, N)..., nc, batchsize) |> device
    Y = randn(Float32, n*ones(Int, N)..., nc, batchsize) |> device
    @test X ≈ C.inverse(C.forward(X)) rtol=1f-6
    @test Y ≈ C.forward(C.inverse(Y)) rtol=1f-6


    # Test backward/inverse coherence
    ΔY = randn(Float32, n*ones(Int, N)..., nc, batchsize) |> device
    Y  = randn(Float32, n*ones(Int, N)..., nc, batchsize) |> device
    X_ = C.inverse(Y)
    _, X = C.backward(ΔY, Y)
    @test X ≈ X_ rtol=1f-6


    # Test backward_inv/forward coherence
    ΔX = randn(Float32, n*ones(Int, N)..., nc, batchsize) |> device
    X  = randn(Float32, n*ones(Int, N)..., nc, batchsize) |> device
    Y_ = C.forward(X)
    _, Y = C.backward_inv(ΔX, X)
    @test Y ≈ Y_ rtol=1f-6


    # Gradient test (input)
    ΔY = randn(Float32, n*ones(Int, N)..., nc, batchsize) |> device
    ΔX = randn(Float32, n*ones(Int, N)..., nc, batchsize) |> device
    X  = randn(Float32, n*ones(Int, N)..., nc, batchsize) |> device
    Y = C.forward(X)
    ΔX_, _ = C.backward(ΔY, Y)
    @test dot(ΔX, ΔX_) ≈ dot(C.forward(ΔX), ΔY) rtol=1f-4


    # Gradient test (input, inv)
    ΔY = randn(Float32, n*ones(Int, N)..., nc, batchsize) |> device
    ΔX = randn(Float32, n*ones(Int, N)..., nc, batchsize) |> device
    Y = randn(Float32, n*ones(Int, N)..., nc, batchsize) |> device
    X = C.inverse(Y)
    ΔY_, _ = C.backward_inv(ΔX, X)
    @test dot(ΔY, ΔY_) ≈ dot(C.inverse(ΔY), ΔX) rtol=1f-4


    # Gradient test (parameters)
    T = Float64
    C = OrthogonalConv1x1(nc; logdet=false) |> device; C.stencil_pars.data = T.(C.stencil_pars.data)
    X  = randn(T, n*ones(Int, N)..., nc, batchsize) |> device; X = T.(X)
    ΔY_ = randn(T, n*ones(Int, N)..., nc, batchsize) |> device; ΔY_ = T.(ΔY_)
    θ = copy(C.stencil_pars.data)
    Δθ = randn(T, size(θ)) |> device; Δθ = T.(Δθ); Δθ *= norm(θ)/norm(Δθ)

    t = T(1e-5)
    set_params!(C, [Parameter(θ+t*Δθ/2)])
    Yp1 = C.forward(X)
    set_params!(C, [Parameter(θ-t*Δθ/2)])
    Ym1 = C.forward(X)
    ΔY = (Yp1-Ym1)/t
    set_params!(C, [Parameter(θ)])
    Y = C.forward(X)
    C.backward(ΔY_, Y)
    Δθ_ = C.stencil_pars.grad

    @test dot(ΔY, ΔY_) ≈ dot(Δθ, Δθ_) rtol=T(1e-4)


    # Gradient test (parameters, inv)
    T = Float64
    Crev = reverse(OrthogonalConv1x1(nc; logdet=false)) |> device; Crev.stencil_pars.data = InvertibleNetworks.CUDA.cu(Crev.stencil_pars.data)
    Y   = InvertibleNetworks.CUDA.randn(T, n*ones(Int, N)..., nc, batchsize)
    ΔX_ = InvertibleNetworks.CUDA.randn(T, n*ones(Int, N)..., nc, batchsize)
    θ = deepcopy(Crev.stencil_pars.data)
    Δθ = InvertibleNetworks.CUDA.randn(T, size(θ)); Δθ *= norm(θ)/norm(Δθ)

    t = T(1e-5)
    set_params!(Crev, [Parameter(θ+t*Δθ/2)])
    Xp1 = Crev.forward(Y)
    set_params!(Crev, [Parameter(θ-t*Δθ/2)])
    Xm1 = Crev.forward(Y)
    ΔX = (Xp1-Xm1)/t
    set_params!(Crev, [Parameter(θ)])
    X = Crev.forward(Y)
    Crev.backward(ΔX_, X)
    Δθ_ = Crev.stencil_pars.grad

    @test dot(ΔX, ΔX_) ≈ dot(Δθ, Δθ_) rtol=T(1e-4)

end