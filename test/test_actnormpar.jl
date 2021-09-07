using GlowInvertibleNetwork, InvertibleNetworks, CUDA, Flux, Test, LinearAlgebra
CUDA.allowscalar(false)
include("./test_utils.jl")

T = Float64

nc = 4
logdet = true
A = ActNormPar(nc; logdet=logdet, T=T)

# Eval
nx = 64
ny = 64
nb = 4
X = randn(T, nx, ny, nc, nb)
Y = A.forward(X)[1]

# Inverse test
X = randn(T, nx, ny, nc, nb)
A.logdet && (@test X ≈ A.inverse(A.forward(X)[1]) rtol=T(1e-5))
~A.logdet && (@test X ≈ A.inverse(A.forward(X)) rtol=T(1e-5))
Y = randn(T, nx, ny, nc, nb)
A.logdet && (@test Y ≈ A.forward(A.inverse(Y))[1] rtol=T(1e-5))
~A.logdet && (@test Y ≈ A.forward(A.inverse(Y)) rtol=T(1e-5))

# Gradient test
loss(X::AbstractArray{T,4}) where T = T(0.5)*norm(X)^2, X
step = T(1e-6)
rtol = T(1e-5)
gradient_test_input(A, loss, X; step=step, rtol=rtol, invnet=true)
gradient_test_pars(A, loss, X; step=step, rtol=rtol, invnet=true)

# Forward (CPU vs GPU)
A = ActNormPar(nc; logdet=logdet, T=Float32)
cpu_vs_gpu_test(A, size(X); rtol=1f-2)