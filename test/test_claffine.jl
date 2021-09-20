using GlowInvertibleNetwork, InvertibleNetworks, CUDA, Flux, Test, LinearAlgebra
CUDA.allowscalar(false)
include("./test_utils.jl")

T = Float64

nc = 4
nc_hidden = 5

k = 3
p = 1
s = 1
actnorm = true
# actnorm = false
weight_std = 0.05
logscale_factor = 3.0
# init_zero = true
init_zero = false
opt_cb = ConvolutionalBlockOptions(; k1=k, p1=p, s1=s, actnorm1=actnorm, k2=k, p2=p, s2=s, actnorm2=actnorm, k3=k, p3=p, s3=s, weight_std1=weight_std, weight_std2=weight_std, weight_std3=weight_std, logscale_factor=logscale_factor, init_zero=init_zero, T=T)
affine = true
# affine = false
α = 0.0
opt = CouplingLayerAffineOptions(; options_convblock=opt_cb, activation=SigmoidNewLayer(T(α)), affine=affine)
logdet = true
# logdet = false
N = CouplingLayerAffine(nc, nc_hidden; logdet=logdet, opt=opt)

# Eval
nx = 64
ny = 64
nb = 4
X = randn(T, nx, ny, nc, nb)
Y = N.forward(X)[1]

# Inverse test
X = randn(T, nx, ny, nc, nb)
N.logdet && (@test X ≈ N.inverse(N.forward(X)[1]) rtol=T(1e-5))
~N.logdet && (@test X ≈ N.inverse(N.forward(X)) rtol=T(1e-5))
Y = randn(T, nx, ny, nc, nb)
N.logdet && (@test Y ≈ N.forward(N.inverse(Y))[1] rtol=T(1e-5))
~N.logdet && (@test Y ≈ N.forward(N.inverse(Y)) rtol=T(1e-5))

# Gradient test
loss(X::AbstractArray{T,4}) where T = T(0.5)*norm(X)^2, X
step = T(1e-5)
rtol = T(1e-3)
gradient_test_input(N, loss, X; step=step, rtol=rtol, invnet=true)
gradient_test_pars(N, loss, X; step=step, rtol=rtol, invnet=true)

# Forward (CPU vs GPU)
opt_cb = ConvolutionalBlockOptions(; k1=k, p1=p, s1=s, actnorm1=actnorm, k2=k, p2=p, s2=s, actnorm2=actnorm, k3=k, p3=p, s3=s, weight_std1=weight_std, weight_std2=weight_std, weight_std3=weight_std, logscale_factor=logscale_factor, init_zero=init_zero, T=Float32)
opt = CouplingLayerAffineOptions(; options_convblock=opt_cb, activation=SigmoidNewLayer(Float32(α)), affine=affine)
N = CouplingLayerAffine(nc, nc_hidden; logdet=logdet, opt=opt)
cpu_vs_gpu_test(N, size(X); rtol=1f-4)