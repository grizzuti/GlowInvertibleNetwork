using GlowInvertibleNetwork, InvertibleNetworks, CUDA, Flux, Test, LinearAlgebra
CUDA.allowscalar(false)
include("./test_utils.jl")
using Random; Random.seed!(1)

T = Float64

nc = 2
nc_hidden = 2
logdet = true
cl_id = false
N = LayerGlow(nc, nc_hidden; logdet=logdet, T=T, cl_id=cl_id)

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
step = T(1e-4)
rtol = T(1e-3)
gradient_test_input(N, loss, X; step=step, rtol=rtol, invnet=true)

θ = get_params(N)
dθ = Array{Parameter,1}(undef, length(θ))
for i = 1:length(θ)
    dθ[i] = Parameter(randn(T, size(θ[i].data)))
    norm(θ[i].data) != T(0) && (dθ[i].data .*= norm(θ[i].data)/norm(dθ[i].data))
    # dθ[i].data .*= norm(θ[i].data)/norm(dθ[i].data)
end
gradient_test_pars(N, loss, X; step=step, rtol=rtol, invnet=true, dθ=dθ)

# Forward (CPU vs GPU)
N = LayerGlow(nc, nc_hidden; logdet=logdet, T=Float32, cl_id=true)
cpu_vs_gpu_test(N, size(X); rtol=1f-4, invnet=true)