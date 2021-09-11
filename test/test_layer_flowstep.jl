using GlowInvertibleNetwork, InvertibleNetworks, CUDA, Flux, Test, LinearAlgebra
CUDA.allowscalar(false)
include("./test_utils.jl")
using Random; Random.seed!(1)

T = Float64

nc = 2
nc_hidden = 3
logdet = true
# cl_id = true
cl_id = false
# conv_orth = true
conv_orth = false
# conv_id = true
conv_id = false
cl_affine = true
# cl_affine = false
N = FlowStep(nc, nc_hidden; logdet=logdet, T=T, cl_activation=SigmoidNewLayer(T(0.5)), cl_affine=cl_affine, cl_id=cl_id, conv_orth=conv_orth)

# Eval
nx = 16
ny = 16
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
N = FlowStep(nc, nc_hidden; logdet=logdet, T=Float32, cl_activation=SigmoidNewLayer(0.5f0), cl_id=cl_id)
cpu_vs_gpu_test(N, size(X); rtol=1f-4, invnet=true)