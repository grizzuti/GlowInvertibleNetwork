using GlowInvertibleNetwork, InvertibleNetworks, LinearAlgebra, Test, Flux, Random
InvertibleNetworks.CUDA.allowscalar(false)
include("./test_utils.jl")
Random.seed!(42)

# Dimensions
n = 32
nc = 4
nc_hidden = 5
batchsize = 3
step = 1e-7
rtol = 1e-4
stencil_size = (3,3,3)
padding = (1,1,1)
stride = (1,1,1)
depth = 3
scales = 2
initial_squeeze = true

device = cpu
# device = gpu

for N = 1:3, do_reverse = [false, true]

    # Test invertibility
    G = Glow(nc; nc_hidden=nc_hidden, stencil_size=stencil_size, padding=padding, stride=stride, init_id_cl=false, depth, scales, logdet=true, initial_squeeze=initial_squeeze, ndims=N) |> device
    InvertibleNetworks.convert_params!(Float64, G)
    X = randn(Float32, n*ones(Int, N)..., nc, batchsize) |> device; X = Float64.(X)
    G.forward(X)
    do_reverse && (G = reverse(G))
    X = randn(Float32, n*ones(Int, N)..., nc, batchsize) |> device; X = Float64.(X)
    Y = randn(Float32, n*ones(Int, N)..., nc, batchsize) |> device; Y = Float64.(Y)
    @test X ≈ G.inverse(G.forward(X)[1]) rtol=1f-5
    @test Y ≈ G.forward(G.inverse(Y))[1] rtol=1f-5

    # Test backward/inverse coherence
    ΔY = randn(Float32, n*ones(Int, N)..., nc, batchsize) |> device; ΔY = Float64.(ΔY)
    Y  = randn(Float32, n*ones(Int, N)..., nc, batchsize) |> device; Y = Float64.(Y)
    X_ = G.inverse(Y)
    _, X = G.backward(ΔY, Y)
    @test X ≈ X_ rtol=1f-6

    # Gradient test (input)
    G = Glow(nc; nc_hidden=nc_hidden, stencil_size=stencil_size, padding=padding, stride=stride, init_id_cl=false, depth, scales, logdet=true, initial_squeeze=initial_squeeze, ndims=N) |> device
    InvertibleNetworks.convert_params!(Float64, G)
    X = randn(Float32, n*ones(Int, N)..., nc, batchsize) |> device; X = Float64.(X)
    G.forward(X)
    do_reverse && (G = reverse(G))
    X  = randn(Float32, n*ones(Int, N)..., nc, batchsize) |> device; X = Float64.(X)
    Y = randn(Float32, n*ones(Int, N)..., nc, batchsize) |> device; Y = Float64.(Y)
    loss(X) = (norm(X-Y)^2/2, X-Y)
    gradient_test_input(G, X; loss=loss, step=step, rtol=rtol, invnet=true)

    # Gradient test (parameters)    
    G = Glow(nc; nc_hidden=nc_hidden, stencil_size=stencil_size, padding=padding, stride=stride, init_id_cl=false, depth, scales, logdet=true, initial_squeeze=initial_squeeze, ndims=N) |> device
    InvertibleNetworks.convert_params!(Float64, G)
    X  = randn(Float32, n*ones(Int, N)..., nc, batchsize) |> device; X = Float64.(X)
    G.forward(X)
    do_reverse && (G = reverse(G))
    X = randn(Float32, n*ones(Int, N)..., nc, batchsize) |> device; X = Float64.(X)
    gradient_test_pars(G, X; loss=loss, step=step, rtol=rtol, invnet=true)

end