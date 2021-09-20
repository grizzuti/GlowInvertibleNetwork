using GlowInvertibleNetwork, InvertibleNetworks, CUDA, Flux, Test, LinearAlgebra
CUDA.allowscalar(false)
include("./test_utils.jl")

T = Float64

nc_in = 4
nc_out = 5
# nc_out = nc_in
nc_hidden = 21
k = 3
p = 1
s = 1
actnorm = true
# actnorm = false
weight_std = 0.05
logscale_factor = 3.0
# init_zero = true
init_zero = false

opt = ConvolutionalBlockOptions(; k1=k, p1=p, s1=s, actnorm1=actnorm, k2=k, p2=p, s2=s, actnorm2=actnorm, k3=k, p3=p, s3=s, weight_std1=weight_std, weight_std2=weight_std, weight_std3=weight_std, logscale_factor=logscale_factor, init_zero=init_zero, T=T)
N = ConvolutionalBlock(nc_in, nc_out, nc_hidden; opt=opt)

# Eval
nx = 64
ny = 64
nb = 4
X = randn(T, nx, ny, nc_in, nb)
Y = N.forward(X)

# Gradient test
loss(X::AbstractArray{T,4}) where T = T(0.5)*norm(X)^2, X
step = T(1e-5)
rtol = T(1e-4)
gradient_test_input(N, loss, X; step=step, rtol=rtol, invnet=false)
gradient_test_pars(N, loss, X; step=step, rtol=rtol, invnet=false)

# Forward (CPU vs GPU)
opt = ConvolutionalBlockOptions(; k1=k, p1=p, s1=s, actnorm1=actnorm, k2=k, p2=p, s2=s, actnorm2=actnorm, k3=k, p3=p, s3=s, weight_std1=weight_std, weight_std2=weight_std, weight_std3=weight_std, logscale_factor=logscale_factor, init_zero=init_zero, T=Float32)
N = ConvolutionalBlock(nc_in, nc_out, nc_hidden; opt=opt)
cpu_vs_gpu_test(N, size(X); rtol=1f-3, invnet=false)