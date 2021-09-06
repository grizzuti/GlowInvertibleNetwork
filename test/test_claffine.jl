using GlowInvertibleNetwork, InvertibleNetworks, CUDA, Flux, Test, LinearAlgebra
CUDA.allowscalar(false)
include("./test_utils.jl")

T = Float64
# T = Float32

nc = 4
nc_hidden = 16
logdet = true
CL = CouplingLayerAffine(nc, nc_hidden; logdet=logdet, T=T, init_id=false)

# Eval
nx = 64
ny = 64
nb = 4
X = randn(T, nx, ny, nc, nb)
Y = CL.forward(X)[1]

# Inverse test
X = randn(T, nx, ny, nc, nb)
CL.logdet && (@test X ≈ CL.inverse(CL.forward(X)[1]) rtol=T(1e-5))
~CL.logdet && (@test X ≈ CL.inverse(CL.forward(X)) rtol=T(1e-5))
Y = randn(T, nx, ny, nc, nb)
CL.logdet && (@test Y ≈ CL.forward(CL.inverse(Y))[1] rtol=T(1e-5))
~CL.logdet && (@test Y ≈ CL.forward(CL.inverse(Y)) rtol=T(1e-5))

# Gradient test
loss(X::AbstractArray{T,4}) where T = T(0.5)*norm(X)^2, X
step = T(1e-6)
rtol = T(1e-5)
gradient_test_input(CL, loss, X; step=step, rtol=rtol, invnet=true)
gradient_test_pars(CL, loss, X; step=step, rtol=rtol, invnet=true)

# Forward (CPU vs GPU)
CL = CouplingLayerAffine(nc, nc_hidden; logdet=logdet, T=Float32)
cpu_vs_gpu_test(CL, size(X); rtol=1f-5)