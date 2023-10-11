using GlowInvertibleNetwork, InvertibleNetworks, CUDA, Flux, Test, LinearAlgebra, Random
CUDA.allowscalar(false)
include("./test_utils.jl")
Random.seed!(42)

T = Float64

n = 64
nc_in = 4
nc_hidden = 21
nc_out = 5
nb = 4
opt = ConvolutionalBlockOptions(; stencil_size=(3,1,3),
                                  padding=(1,0,1),
                                  stride=(1,1,1),
                                  do_actnorm=true,
                                  init_zero=false)
step = T(1e-6)
rtol = T(1e-3)
for ndims = [2, 3]

    # Initialize
    N = ConvolutionalBlock(nc_in, nc_hidden, nc_out; opt=opt, ndims=ndims) |> gpu
    InvertibleNetworks.convert_params!(T, N)

    # Eval
    X = randn(T, n*ones(Int, ndims)..., nc_in, nb) |> gpu
    X = convert.(T, X)
    Y = N.forward(X)::AbstractArray{T}
    X = randn(T, n*ones(Int, ndims)..., nc_in, nb) |> gpu
    X = convert.(T, X)

    # Gradient test
    loss(X::AbstractArray{T}) where T = norm(X)^2/2, X
    gradient_test_input(N, loss, X; step=step, rtol=rtol, invnet=false)
    gradient_test_pars(N, loss, X; step=step, rtol=rtol, invnet=false)

end