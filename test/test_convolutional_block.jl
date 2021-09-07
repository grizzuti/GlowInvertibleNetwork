using GlowInvertibleNetwork, InvertibleNetworks, CUDA, Flux, Test, LinearAlgebra
CUDA.allowscalar(false)
include("./test_utils.jl")

T = Float64

nc_in = 4
nc_out = 5
nc_hidden = 21
k = 3
p = 1
s = 1
bias = true
weight_std = 0.05
logscale_factor = 3.0
N = ConvolutionalBlock(nc_in, nc_out, nc_hidden; k1=k, p1=p, s1=s, actnorm1=true, k2=k, p2=p, s2=s, actnorm2=true, k3=k, p3=p, s3=s, weight_std1=weight_std, weight_std2=weight_std, logscale_factor=logscale_factor, T=T, init_zero=false)

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
N = ConvolutionalBlock(nc_in, nc_out, nc_hidden; k1=k, p1=p, s1=s, actnorm1=true, k2=k, p2=p, s2=s, actnorm2=true, k3=k, p3=p, s3=s, weight_std1=weight_std, weight_std2=weight_std, logscale_factor=logscale_factor, T=Float32, init_zero=false)
cpu_vs_gpu_test(N, size(X); rtol=1f-4, invnet=false)