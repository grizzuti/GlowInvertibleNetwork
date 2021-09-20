using GlowInvertibleNetwork, InvertibleNetworks, CUDA, Flux, Test, LinearAlgebra
CUDA.allowscalar(false)
include("./test_utils.jl")
# using Random; Random.seed!(2)

T = Float64

α = 0.5
cl_affine = true
# cl_affine = false
# init_cl_id = true
init_cl_id = false
conv1x1_nvp = true
# conv1x1_nvp = false
init_conv1x1_permutation = true
# init_conv1x1_permutation = false
opt = GlowLowDimOptions(; cl_activation=SigmoidNewLayer(T(α)),
                    cl_affine=cl_affine,
                    init_cl_id=init_cl_id,
                    conv1x1_nvp=conv1x1_nvp,
                    init_conv1x1_permutation=init_conv1x1_permutation,
                    T=T)
nc = 2
nc_hidden = 3
depth = 2
logdet = true
# logdet = false
N = GlowLowDim(nc, nc_hidden, depth; logdet=logdet, opt=opt)

# Eval
nx = 1
ny = 1
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
opt = GlowLowDimOptions(; cl_activation=SigmoidNewLayer(Float32(α)),
                    cl_affine=cl_affine,
                    init_cl_id=init_cl_id,
                    conv1x1_nvp=conv1x1_nvp,
                    init_conv1x1_permutation=init_conv1x1_permutation,
                    T=Float32)
N = GlowLowDim(nc, nc_hidden, depth; logdet=logdet, opt=opt)
cpu_vs_gpu_test(N, size(X); rtol=1f-4, invnet=true)