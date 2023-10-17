using GlowInvertibleNetwork, InvertibleNetworks, LinearAlgebra, Test, Flux, Random
device = InvertibleNetworks.CUDA.functional() ? gpu : cpu
include("./test_utils.jl")
Random.seed!(11)

# Dimensions
n = 64
nc = 4
nc_hidden = 5
batchsize = 3

for N = 1:3, do_reverse = [false, true]

    # Test invertibility
    CL = CouplingLayerAffine(nc, nc_hidden; activation=SigmoidLayerNew(; low=0.5f0, high=1f0), logdet=true, init_id=false, ndims=N) |> device; do_reverse && (CL = reverse(CL))
    X = randn(Float32, n*ones(Int, N)..., nc, batchsize) |> device
    Y = randn(Float32, n*ones(Int, N)..., nc, batchsize) |> device
    @test X ≈ CL.inverse(CL.forward(X)[1]) rtol=1f-6
    @test Y ≈ CL.forward(CL.inverse(Y))[1] rtol=1f-6


    # Test backward/inverse coherence
    ΔY = randn(Float32, n*ones(Int, N)..., nc, batchsize) |> device
    Y  = randn(Float32, n*ones(Int, N)..., nc, batchsize) |> device
    X_ = CL.inverse(Y)
    _, X = CL.backward(ΔY, Y)
    @test X ≈ X_ rtol=1f-6


    # Gradient test (input)
    CL = CouplingLayerAffine(nc, nc_hidden; activation=SigmoidLayerNew(; low=0.5, high=1.0), logdet=true, init_id=false, ndims=N) |> device; do_reverse && (CL = reverse(CL))
    θ = get_params(CL); for i = eachindex(θ) ~isnothing(θ[i].data) && (θ[i].data = Float64.(θ[i].data)); end
    ΔY = randn(Float32, n*ones(Int, N)..., nc, batchsize) |> device; ΔY = Float64.(ΔY)
    ΔX = randn(Float32, n*ones(Int, N)..., nc, batchsize) |> device; ΔX = Float64.(ΔX)
    X  = randn(Float32, n*ones(Int, N)..., nc, batchsize) |> device; X = Float64.(X)
    CL.forward(X)
    X  = randn(Float32, n*ones(Int, N)..., nc, batchsize) |> device; X = Float64.(X)
    loss(X::AbstractArray{T}) where T = norm(X)^2/2, X
    step = 1e-7
    rtol = 1e-3
    gradient_test_input(CL, loss, X; step=step, rtol=rtol, invnet=true)


    # Gradient test (parameters)    
    CL = CouplingLayerAffine(nc, nc_hidden; activation=SigmoidLayerNew(; low=0.5, high=1.0), logdet=true, init_id=false, ndims=N) |> device; do_reverse && (CL = reverse(CL))
    θ = get_params(CL); for i = eachindex(θ) ~isnothing(θ[i].data) && (θ[i].data = Float64.(θ[i].data)); end
    X  = randn(Float32, n*ones(Int, N)..., nc, batchsize) |> device; X = Float64.(X)
    CL.forward(X)
    X  = randn(Float32, n*ones(Int, N)..., nc, batchsize) |> device; X = Float64.(X)
    gradient_test_pars(CL, loss, X; step=step, rtol=rtol, invnet=true)

end