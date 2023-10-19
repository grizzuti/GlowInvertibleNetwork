using GlowInvertibleNetwork, InvertibleNetworks, LinearAlgebra, Test, Flux, Random
InvertibleNetworks.CUDA.allowscalar(false)
include("./test_utils.jl")
Random.seed!(42)

# Dimensions
n = 16
nc = 4
nc_hidden = 5
batchsize = 3
step = 1e-6
rtol = 1e-4
stencil_size = (3,3,3)
padding = (1,1,1)
stride = (1,1,1)

device = cpu
# device = gpu

for N = 1:3, do_reverse = [false, true]

    # Test invertibility
    FS = FlowStep(nc; nc_hidden=nc_hidden, stencil_size=stencil_size, padding=padding, stride=stride, logdet=true, init_id_cl=false, ndims=N) |> device
    X = randn(Float32, n*ones(Int, N)..., nc, batchsize) |> device
    FS.forward(X)
    do_reverse && (FS = reverse(FS))
    X = randn(Float32, n*ones(Int, N)..., nc, batchsize) |> device
    Y = randn(Float32, n*ones(Int, N)..., nc, batchsize) |> device
    @test X ≈ FS.inverse(FS.forward(X)[1]) rtol=1f-5
    @test Y ≈ FS.forward(FS.inverse(Y))[1] rtol=1f-5

    # Test backward/inverse coherence
    ΔY = randn(Float32, n*ones(Int, N)..., nc, batchsize) |> device
    Y  = randn(Float32, n*ones(Int, N)..., nc, batchsize) |> device
    X_ = FS.inverse(Y)
    _, X = FS.backward(ΔY, Y)
    @test X ≈ X_ rtol=1f-6

    # Gradient test (input)
    FS = FlowStep(nc; nc_hidden=nc_hidden, stencil_size=stencil_size, padding=padding, stride=stride, logdet=true, init_id_cl=false, ndims=N) |> device
    InvertibleNetworks.convert_params!(Float64, FS)
    X = randn(Float32, n*ones(Int, N)..., nc, batchsize) |> device; X = Float64.(X)
    FS.forward(X)
    do_reverse && (FS = reverse(FS))
    X  = randn(Float32, n*ones(Int, N)..., nc, batchsize) |> device; X = Float64.(X)
    Y = randn(Float32, n*ones(Int, N)..., nc, batchsize) |> device; Y = Float64.(Y)
    loss(X) = (norm(X-Y)^2/2, X-Y)
    gradient_test_input(FS, X; loss=loss, step=step, rtol=rtol, invnet=true)

    # Gradient test (parameters)    
    FS = FlowStep(nc; nc_hidden=nc_hidden, stencil_size=stencil_size, padding=padding, stride=stride, logdet=true, init_id_cl=false, ndims=N) |> device
    InvertibleNetworks.convert_params!(Float64, FS)
    X  = randn(Float32, n*ones(Int, N)..., nc, batchsize) |> device; X = Float64.(X)
    FS.forward(X)
    do_reverse && (FS = reverse(FS))
    X = randn(Float32, n*ones(Int, N)..., nc, batchsize) |> device; X = Float64.(X)
    gradient_test_pars(FS, X; loss=loss, step=step, rtol=rtol, invnet=true)

end