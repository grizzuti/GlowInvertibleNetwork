using GlowInvertibleNetwork, InvertibleNetworks, CUDA, Flux, Test, LinearAlgebra
CUDA.allowscalar(false)
include("./test_utils.jl")

T = Float64
# T = Float32

nc = 4
# logdet = true
logdet = false
C = Conv1x1gen(nc; logdet=logdet, T=T)

# Eval
nx = 64
ny = 64
nb = 4
X = randn(T, nx, ny, nc, nb)
Y = C.forward(X)

# Inverse test
X = randn(T, nx, ny, nc, nb)
C.logdet && (@test X ≈ C.inverse(C.forward(X)[1]) rtol=T(1e-5))
~C.logdet && (@test X ≈ C.inverse(C.forward(X)) rtol=T(1e-5))
Y = randn(T, nx, ny, nc, nb)
C.logdet && (@test Y ≈ C.forward(C.inverse(Y))[1] rtol=T(1e-5))
~C.logdet && (@test Y ≈ C.forward(C.inverse(Y)) rtol=T(1e-5))

# Gradient test
loss(X::AbstractArray{T,4}) where T = T(0.5)*norm(X)^2, X
step = T(1e-6)
rtol = T(1e-5)
gradient_test_input(C, loss, X; step=step, rtol=rtol, invnet=true)
gradient_test_pars(C, loss, X; step=step, rtol=rtol, invnet=true)

# Forward (CPU vs GPU)
C = Conv1x1gen(nc; logdet=false, T=Float32)
cpu_vs_gpu_test(C, size(X); rtol=1f-5)