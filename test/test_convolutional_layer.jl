using GlowInvertibleNetwork, InvertibleNetworks, Flux, Test, LinearAlgebra
include("./test_utils.jl")

T = Float64

n = 64
nc_in = 4
nc_out = 5
nb = 4
k = 3
p = 1
s = 1
bias = true
init_zero = false

for ndims = 1:3

    # Initialize
    N = ConvolutionalLayer(nc_in, nc_out; stencil_size=k, padding=p, stride=s, bias=bias, init_zero=init_zero, ndims=ndims) |> gpu
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