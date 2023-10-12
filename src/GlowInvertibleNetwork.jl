module GlowInvertibleNetwork

using LinearAlgebra, NNlib, Flux, InvertibleNetworks, Statistics

include("./convolutional_layer.jl")
include("./convolutional_block.jl")
include("./invertible_layer_orthogonalconv1x1.jl")
# include("./invertible_layer_claffine.jl")
# include("./invertible_layer_flowstep.jl")
# include("./invertible_network_glow.jl")
# include("./invertible_network_glowlowdim.jl")

end