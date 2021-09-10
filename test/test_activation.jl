using GlowInvertibleNetwork, InvertibleNetworks, CUDA, Flux, Test, LinearAlgebra
CUDA.allowscalar(false)
include("./test_utils.jl")
using Random; Random.seed!(1)

T = Float64

# Eval
nx = 16
ny = 16
nc = 3
nb = 4
X = randn(T, nx, ny, nc, nb)
Y = ExpClampNew(X)

# Gradient test
step = 1e-4
tol = 1e-5
loss(X::AbstractArray{T,4}) where T = T(0.5)*norm(X)^2, X
_, ΔY = loss(Y)
dX = ExpClampNewGrad(ΔY, X)
step = T(1e-5)
rtol = T(1e-4)
ΔX = randn(T, nx, ny, nc, nb); ΔX *= norm(X)/norm(ΔX)
Yp1 = ExpClampNew(X+0.5*step*ΔX)
lp1, _ = loss(Yp1)
Ym1 = ExpClampNew(X-0.5*step*ΔX)
lm1, _ = loss(Ym1)
@test (lp1-lm1)/step ≈ dot(dX, ΔX) rtol=rtol