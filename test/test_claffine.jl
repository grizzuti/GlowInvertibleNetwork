using GlowInvertibleNetwork, InvertibleNetworks, LinearAlgebra, Test, Flux, Random
include("./test_utils.jl")
Random.seed!(42)

# Dimensions
n = 16
nc = 4
nc_hidden = 5
batchsize = 3
step = 1e-6
rtol = 1e-4

device = cpu
# device = gpu

for N = 1:3, do_reverse = [false, true], init_id = [false, true]

    # Test invertibility
    CL = CouplingLayerAffine(nc; nc_hidden=nc_hidden, activation=ExpClampLayerNew(; clamp=2), logdet=true, init_id=init_id, ndims=N) |> device; do_reverse && (CL = reverse(CL))
    X = randn(Float32, n*ones(Int, N)..., nc, batchsize) |> device
    Y = randn(Float32, n*ones(Int, N)..., nc, batchsize) |> device
    @test X ≈ CL.inverse(CL.forward(X)[1]) rtol=1f-6
    @test Y ≈ CL.forward(CL.inverse(Y))[1] rtol=1f-6

    # Test identity
    if init_id
        X = randn(Float32, n*ones(Int, N)..., nc, batchsize) |> device
        @test CL.forward(X)[1] ≈ X rtol=1f-6
    end

    # Test backward/inverse coherence
    ΔY = randn(Float32, n*ones(Int, N)..., nc, batchsize) |> device
    Y  = randn(Float32, n*ones(Int, N)..., nc, batchsize) |> device
    X_ = CL.inverse(Y)
    _, X = CL.backward(ΔY, Y)
    @test X ≈ X_ rtol=1f-6

    # Gradient test (input)
    CL = CouplingLayerAffine(nc; nc_hidden=nc_hidden, activation=ExpClampLayerNew(; clamp=2), logdet=true, init_id=init_id, ndims=N) |> device; do_reverse && (CL = reverse(CL))
    InvertibleNetworks.convert_params!(Float64, CL)
    ΔY = randn(Float32, n*ones(Int, N)..., nc, batchsize) |> device; ΔY = Float64.(ΔY)
    ΔX = randn(Float32, n*ones(Int, N)..., nc, batchsize) |> device; ΔX = Float64.(ΔX)
    X  = randn(Float32, n*ones(Int, N)..., nc, batchsize) |> device; X = Float64.(X)
    CL.forward(X)
    X  = randn(Float32, n*ones(Int, N)..., nc, batchsize) |> device; X = Float64.(X)
    gradient_test_input(CL, X; step=step, rtol=rtol, invnet=true)

    # Gradient test (parameters)    
    CL = CouplingLayerAffine(nc; nc_hidden=nc_hidden, activation=ExpClampLayerNew(; clamp=2), logdet=true, init_id=init_id, ndims=N) |> device; do_reverse && (CL = reverse(CL))
    InvertibleNetworks.convert_params!(Float64, CL)
    X  = randn(Float32, n*ones(Int, N)..., nc, batchsize) |> device; X = Float64.(X)
    CL.forward(X)
    X  = randn(Float32, n*ones(Int, N)..., nc, batchsize) |> device; X = Float64.(X)
    gradient_test_pars(CL, X; step=step, rtol=rtol, invnet=true)

end