module GlowInvertibleNetwork

using LinearAlgebra, CUDA, Flux, InvertibleNetworks, Statistics
import Flux: gpu, cpu
import InvertibleNetworks: forward, inverse, backward, clear_grad!, get_params

include("./parameter_tricks.jl")
include("./convolutional_layer.jl")
include("./convolutional_layer0.jl")
include("./actnorm_par.jl")
include("./convolutional_block.jl")
include("./invertible_layer_conv1x1gen.jl")

end
