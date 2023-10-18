using GlowInvertibleNetwork, InvertibleNetworks, Flux, Test, LinearAlgebra, Random
InvertibleNetworks.CUDA.allowscalar(false)
include("./test_utils.jl")
Random.seed!(42)

n = 16
nc_in = 4
nc_hidden = 8
nc_out = 6
nb = 4
stencil_size = (3,3,3)
padding = (1,1,1)
stride = (1,1,1)
do_actnorm = true
step = 1e-6
rtol = 1e-3

device = cpu
# device = gpu

for N = 1:3, init_zero = [false, true]

    # Initialize
    CB = ConvolutionalBlock(nc_in, nc_hidden, nc_out;
                                stencil_size=stencil_size,
                                padding=padding,
                                stride=stride,
                                do_actnorm=do_actnorm,
                                init_zero=init_zero,
                                ndims=N) |> device
    InvertibleNetworks.convert_params!(Float64, CB)

    # Zero test
    if init_zero
        X = randn(Float64, n*ones(Int, N)..., nc_in, nb) |> device; X = Float64.(X)
        @test norm(CB.forward(X)) ≈ 0
    end

    # Gradient test
    if ~init_zero
        X = randn(Float64, n*ones(Int, N)..., nc_in, nb) |> device; X = Float64.(X)
        gradient_test_input(CB, X; step=step, rtol=rtol, invnet=false)
        gradient_test_pars(CB, X; step=step, rtol=rtol, invnet=false)
    end

end