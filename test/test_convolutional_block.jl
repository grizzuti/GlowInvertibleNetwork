using GlowInvertibleNetwork, InvertibleNetworks, Flux, Test, LinearAlgebra
include("./test_utils.jl")

nc_in = 4
nc_out = 5
nc_hidden = 21
k = 3
p = 1
s = 1
bias = true
weight_std = 0.05
logscale_factor = 3.0
T = Float64
L = ConvolutionalBlock(nc_in, nc_out, nc_hidden; k1=k, p1=p, s1=s, actnorm1=true, k2=k, p2=p, s2=s, actnorm2=true, k3=k, p3=p, s3=s, weight_std1=weight_std, weight_std2=weight_std, logscale_factor=logscale_factor, T=T)
L.CL3.CL.W.data = randn(T,k,k,nc_hidden,nc_out)
L.CL3.CL.b.data = randn(T,1,1,nc_out,1)

# Eval
nx = 64
ny = 64
nb = 4
X = randn(T, nx, ny, nc_in, nb)
Y = L.forward(X)

# Gradient test
loss(X::AbstractArray{T,4}) where T = T(0.5)*norm(X)^2, X
gradient_test_input(L, loss, X; step=T(1e-6), rtol=T(1e-5), invnet=false)
gradient_test_pars(L, loss, X; step=T(1e-6), rtol=T(1e-5), invnet=false)