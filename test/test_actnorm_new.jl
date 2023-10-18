using GlowInvertibleNetwork, InvertibleNetworks, LinearAlgebra, Test, Flux, Random
include("./test_utils.jl")
Random.seed!(42)

# Dimensions
n = 16
nc = 4
batchsize = 3
step = 1e-6
rtol = 1e-5

device = cpu
# device = gpu

for N = 1:3, do_reverse = [false, true], init_id = [false, true]

    # Test invertibility
    AN = ActNormNew(; logdet=true, init_id=init_id) |> device
    X = randn(Float32, n*ones(Int, N)..., nc, batchsize) |> device
    AN.forward(X)
    do_reverse && (AN = reverse(AN))
    X = randn(Float32, n*ones(Int, N)..., nc, batchsize) |> device
    Y = randn(Float32, n*ones(Int, N)..., nc, batchsize) |> device
    @test X ≈ AN.inverse(AN.forward(X)[1]) rtol=1f-6
    @test Y ≈ AN.forward(AN.inverse(Y))[1] rtol=1f-6

    # Identity test
    if init_id
        X = randn(Float32, n*ones(Int, N)..., nc, batchsize) |> device
        @test AN.forward(X)[1] ≈ X
    end

    # Test backward/inverse coherence
    ΔY = randn(Float32, n*ones(Int, N)..., nc, batchsize) |> device
    Y  = randn(Float32, n*ones(Int, N)..., nc, batchsize) |> device
    X_ = AN.inverse(Y)
    _, X = AN.backward(ΔY, Y)
    @test X ≈ X_ rtol=1f-6

    # Gradient test (input)
    AN = ActNormNew(; logdet=true, init_id=init_id) |> device
    X = randn(Float32, n*ones(Int, N)..., nc, batchsize) |> device
    AN.forward(X)
    do_reverse && (AN = reverse(AN))
    InvertibleNetworks.convert_params!(Float64, AN)
    ΔY = randn(Float32, n*ones(Int, N)..., nc, batchsize) |> device; ΔY = Float64.(ΔY)
    ΔX = randn(Float32, n*ones(Int, N)..., nc, batchsize) |> device; ΔX = Float64.(ΔX)
    X = randn(Float32, n*ones(Int, N)..., nc, batchsize) |> device; X = Float64.(X)
    gradient_test_input(AN, X; step=step, rtol=rtol, invnet=true)

    # Gradient test (parameters)    
    AN = ActNormNew(; logdet=true, init_id=init_id) |> device
    X = randn(Float32, n*ones(Int, N)..., nc, batchsize) |> device
    AN.forward(X)
    do_reverse && (AN = reverse(AN))    
    InvertibleNetworks.convert_params!(Float64, AN)
    X = randn(Float32, n*ones(Int, N)..., nc, batchsize) |> device; X = Float64.(X)
    gradient_test_pars(AN, X; step=step, rtol=rtol, invnet=true)

end