module GlowInvertibleNetwork

using LinearAlgebra, CUDA, Flux, InvertibleNetworks, Statistics

include("./convolutional_layer.jl")
include("./convolutional_layer0.jl")
# include("./invertible_layer_actnorm_par.jl")
# include("./convolutional_block.jl")
# include("./invertible_layer_conv1x1gen.jl")
# include("./invertible_layer_conv1x1orth_fixed.jl")
# include("./invertible_layer_claffine.jl")
# include("./invertible_layer_flowstep.jl")
# include("./invertible_network_glow.jl")
# include("./invertible_network_glowlowdim.jl")

end