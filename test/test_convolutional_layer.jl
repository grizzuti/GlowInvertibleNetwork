using GlowInvertibleNetwork, InvertibleNetworks, CUDA, Flux, Test, LinearAlgebra
CUDA.allowscalar(false)
include("./test_utils.jl")

T = Float64

nc_in = 4
nc_out = 5
k = 3
p = 1
s = 1
bias = true
weight_std = 0.05
logscale_factor = 3.0
N = ConvolutionalLayer(nc_in, nc_out; k=k, p=p, s=s, bias=bias, weight_std=weight_std, T=T)

# Eval
nx = 64
ny = 64
nb = 4
X = randn(T, nx, ny, nc_in, nb)
Y = N.forward(X)[1]

# Gradient test
loss(X::AbstractArray{T,4}) where T = T(0.5)*norm(X)^2, X
step = T(1e-6)
rtol = T(1e-5)
gradient_test_input(N, loss, X; step=step, rtol=rtol, invnet=false)
gradient_test_pars(N, loss, X; step=step, rtol=rtol, invnet=false)

# Forward (CPU vs GPU)
N = ConvolutionalLayer(nc_in, nc_out; k=k, p=p, s=s, bias=bias, weight_std=weight_std, T=Float32)
cpu_vs_gpu_test(N, size(X); rtol=1f-4, invnet=false)