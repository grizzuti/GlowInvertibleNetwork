using GlowInvertibleNetwork, InvertibleNetworks, Flux, Test, LinearAlgebra
include("./test_utils.jl")

n = 16
nc_in = 4
nc_out = 5
nb = 4
step = 1e-6
rtol = 1e-5

device = cpu
# device = gpu

for N = 1:3, bias = [true, false], init_zero = [false, true]

    # Initialize
    CL = ConvolutionalLayer(nc_in, nc_out; stencil_size=3, padding=1, stride=1, bias=bias, init_zero=init_zero, ndims=N) |> device
    InvertibleNetworks.convert_params!(Float64, CL)

    # Zero test
    if init_zero
        X = randn(Float64, n*ones(Int, N)..., nc_in, nb) |> device; X = Float64.(X)
        @test norm(CL.forward(X)) ≈ 0
    end

    # Gradient test
    X = randn(Float64, n*ones(Int, N)..., nc_in, nb) |> device; X = Float64.(X)
    gradient_test_input(CL, X; step=step, rtol=rtol, invnet=false)
    gradient_test_pars(CL, X; step=step, rtol=rtol, invnet=false)

end