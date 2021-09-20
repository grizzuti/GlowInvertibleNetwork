using GlowInvertibleNetwork, InvertibleNetworks, CUDA, Flux, Test, LinearAlgebra
CUDA.allowscalar(false)
include("./test_utils.jl")
using Random; Random.seed!(1)

T = Float64

nc = 4
logdet = true
nvp = true
# nvp = false
init_permutation = true
# init_permutation = false
opt = Conv1x1genOptions(; nvp=nvp, init_permutation=init_permutation, T=T)
N = Conv1x1gen(nc; logdet=logdet, opt=opt)

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
step = T(1e-6)
rtol = T(1e-5)
gradient_test_input(N, loss, X; step=step, rtol=rtol, invnet=true)
gradient_test_pars(N, loss, X; step=step, rtol=rtol, invnet=true)

# Forward (CPU vs GPU)
opt = Conv1x1genOptions(; nvp=nvp, init_permutation=init_permutation, T=Float32)
N = Conv1x1gen(nc; logdet=logdet, opt=opt)
N.l.data = randn(Float32, size(N.l.data))
N.u.data = randn(Float32, size(N.u.data))
N.nvp && (N.s.data = randn(Float32, size(N.s.data)))
cpu_vs_gpu_test(N, size(X); rtol=1f-4)