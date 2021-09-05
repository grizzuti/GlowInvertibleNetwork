using GlowInvertibleNetwork, InvertibleNetworks, Flux, Test, LinearAlgebra
include("./test_utils.jl")

nc = 4
T = Float64
A = ActNormPar(nc; logdet=false, T=T)

# Eval
nx = 64
ny = 64
nb = 3
X = randn(T, nx, ny, nc, nb)
Y = A.forward(X)

# Gradient test
X = randn(T, nx, ny, nc, nb)
loss(X::AbstractArray{T,4}) where T = T(0.5)*norm(X)^2, X
gradient_test_input(A, loss, X; step=T(1e-6), rtol=T(1e-5), invnet=true)
gradient_test_pars(A, loss, X; step=T(1e-6), rtol=T(1e-5), invnet=true)