using GlowInvertibleNetwork, InvertibleNetworks, CUDA, Flux, Test, LinearAlgebra
CUDA.allowscalar(false)
include("./test_utils.jl")
# using Random; Random.seed!(2)

T = Float64

nc = 1
nc_hidden = 3
logdet = true
# logdet = false
depth = 2
nscales = 2
# cl_id = true
cl_id = false
# conv_orth = true
conv_orth = false
# conv_id = true
conv_id = false
cl_affine = true
# cl_affine = false
α = 0.1
N = Glow(nc, nc_hidden, depth, nscales; logdet=logdet, cl_id=cl_id, conv_orth=conv_orth, conv_id=conv_id, cl_affine=cl_affine, cl_activation=SigmoidNewLayer(T(α)), T=T)

# Eval
nx = 64
ny = 64
nb = 4
X = randn(T, nx, ny, nc, nb)
Y = N.forward(X)[1]

# Inverse test
X = randn(T, nx, ny, nc, nb)
N.logdet && (@test X ≈ N.inverse(N.forward(X)[1]) rtol=T(1e-3))
~N.logdet && (@test X ≈ N.inverse(N.forward(X)) rtol=T(1e-3))
Y = randn(T, nx, ny, nc, nb)
N.logdet && (@test Y ≈ N.forward(N.inverse(Y))[1] rtol=T(1e-3))
~N.logdet && (@test Y ≈ N.forward(N.inverse(Y)) rtol=T(1e-3))

# Gradient test
loss(X::AbstractArray{T,4}) where T = T(0.5)*norm(X)^2, X
step = T(1e-5)
rtol = T(1e-3)
gradient_test_input(N, loss, X; step=step, rtol=rtol, invnet=true)
gradient_test_pars(N, loss, X; step=step, rtol=rtol, invnet=true)

# Forward (CPU vs GPU)
N = Glow(nc, nc_hidden, depth, nscales; logdet=logdet, cl_id=cl_id, conv_orth=conv_orth, conv_id=conv_id, cl_affine=cl_affine, cl_activation=SigmoidNewLayer(Float32(α)), T=Float32)
cpu_vs_gpu_test(N, size(X); rtol=1f-4, invnet=true)