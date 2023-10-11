using GlowInvertibleNetwork, CUDA, Test

@testset "GlowInvertibleNetwork.jl" begin
    include("./test_convolutional_layer.jl")
    # include("./test_convolutional_layer0.jl")
    include("./test_convolutional_block.jl")
end