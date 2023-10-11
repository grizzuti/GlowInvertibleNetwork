using GlowInvertibleNetwork, InvertibleNetworks, CUDA, Flux, Test, LinearAlgebra
CUDA.allowscalar(false)
include("./test_utils.jl")

T = Float64

n = 64
nc_in = 4
nc_out = 5
nb = 4
k = 3
p = 1
s = 1
weight_std = 0.05
logscale_factor = 3.0

for ndims = [2, 3]

    # Initialize
    N = ConvolutionalLayer0(nc_in, nc_out; stencil_size=k, padding=p, stride=s, logscale_factor=logscale_factor, weight_std=weight_std, ndims=ndims) |> gpu
    InvertibleNetworks.convert_params!(T, N)

    # Eval
    X = randn(T, n*ones(Int, ndims)..., nc_in, nb) |> gpu
    X = convert.(T, X)
    Y = N.forward(X)::AbstractArray{T}

    # Gradient test
    loss(X::AbstractArray{T}) where T = norm(X)^2/2, X
    step = T(1e-6)
    rtol = T(1e-5)
    gradient_test_input(N, loss, X; step=step, rtol=rtol, invnet=false)
    gradient_test_pars(N, loss, X; step=step, rtol=rtol, invnet=false)

end