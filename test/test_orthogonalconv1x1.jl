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
    Q = OrthogonalConv1x1(nc; logdet=true, init_id=init_id) |> device; do_reverse && (Q = reverse(Q))
    X = randn(Float32, n*ones(Int, N)..., nc, batchsize) |> device
    Y = randn(Float32, n*ones(Int, N)..., nc, batchsize) |> device
    @test X ≈ Q.inverse(Q.forward(X)[1]) rtol=1f-6
    @test Y ≈ Q.forward(Q.inverse(Y))[1] rtol=1f-6

    # Identity test
    if init_id
        X = randn(Float32, n*ones(Int, N)..., nc, batchsize) |> device
        @test Q.forward(X)[1] ≈ X
    end

    # Test backward/inverse coherence
    ΔY = randn(Float32, n*ones(Int, N)..., nc, batchsize) |> device
    Y  = randn(Float32, n*ones(Int, N)..., nc, batchsize) |> device
    X_ = Q.inverse(Y)
    _, X = Q.backward(ΔY, Y)
    @test X ≈ X_ rtol=1f-6

    # Gradient test (input)
    Q = OrthogonalConv1x1(nc; logdet=true, init_id=init_id) |> device; do_reverse && (Q = reverse(Q))
    InvertibleNetworks.convert_params!(Float64, Q)
    X  = randn(Float32, n*ones(Int, N)..., nc, batchsize) |> device; X = Float64.(X)
    Y = randn(Float32, n*ones(Int, N)..., nc, batchsize) |> device; Y = Float64.(Y)
    loss(X) = (norm(X-Y)^2/2, X-Y)
    gradient_test_input(Q, X; loss=loss, step=step, rtol=rtol, invnet=true)

    # Gradient test (parameters)
    Q = OrthogonalConv1x1(nc; logdet=true, init_id=init_id) |> device; do_reverse && (Q = reverse(Q))
    InvertibleNetworks.convert_params!(Float64, Q)
    X = randn(Float32, n*ones(Int, N)..., nc, batchsize) |> device; X = Float64.(X)
    Y = randn(Float32, n*ones(Int, N)..., nc, batchsize) |> device; Y = Float64.(Y)
    gradient_test_pars(Q, X; loss=loss, step=step, rtol=rtol, invnet=true)

end