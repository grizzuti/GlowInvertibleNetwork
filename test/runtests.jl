using GlowInvertibleNetwork, Test

@testset "GlowInvertibleNetwork.jl" begin
    include("./test_sigmoid_new.jl")
    include("./test_actnormnew.jl")
    include("./test_convolutional_layer.jl")
    include("./test_convolutional_block.jl")
    include("./test_orthogonalconv1x1.jl")
    include("./test_claffine.jl")
    include("./test_layer_flowstep.jl")
end