#################################################################################
# Train generator on MRI images
#################################################################################

using LinearAlgebra, InvertibleNetworks, PyPlot, Flux, CUDA, JLD, GlowInvertibleNetwork
import Flux.Optimise: Optimiser, ADAM, ExpDecay, update!
using Random; Random.seed!(1)
include("./plotting_utils.jl")

# Create multiscale network
opt = GlowOptions(; cl_activation=SigmoidNewLayer(0.5f0),
                    cl_affine=true,
                    init_cl_id=true,
                    conv1x1_nvp=false,
                    init_conv1x1_permutation=true,
                    T=Float32)
nc = 2
nc_hidden = 4
depth = 5
nscales = 1
G = Glow(nc, nc_hidden, depth, nscales; opt=opt)

batch_size = 16
X = randn(Float32, 1, 1, nc, batch_size)
G.forward(X)