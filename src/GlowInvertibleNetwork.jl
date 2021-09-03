module GlowInvertibleNetwork

using CUDA, Flux, InvertibleNetworks
import Flux: gpu, cpu
import InvertibleNetworks: forward, inverse, backward, clear_grad!, get_params

include("./parameter_tricks.jl")
include("./convolutional_layer.jl")
include("./convolutional_layer0.jl")
#include("./convolutional_block.jl")

end
