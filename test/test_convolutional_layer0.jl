using GlowInvertibleNetwork, InvertibleNetworks, Flux, Test, LinearAlgebra
include("./test_utils.jl")

nc_in = 4
nc_out = 5
k = 3
p = 1
s = 1
logscale_factor = 3.
T = Float64
L = ConvolutionalLayer0(nc_in, nc_out; k=k, p=p, s=s, logscale_factor=logscale_factor, T=T)
L.CL.W.data = randn(T,k,k,nc_in,nc_out)
L.CL.b.data = randn(T,1,1,nc_out,1)

# Eval
nx = 64
ny = 64
nb = 4
X = randn(T, nx, ny, nc_in, nb)
Y = L.forward(X)

# Gradient test
loss(X::AbstractArray{T,4}) where T = T(0.5)*norm(X)^2, X
gradient_test_input(L, loss, X; step=T(1e-6), rtol=T(1e-5))
gradient_test_pars(L, loss, X; step=T(1e-6), rtol=T(1e-5))