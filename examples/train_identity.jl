#################################################################################
# Train generator on MRI images
#################################################################################

using LinearAlgebra, InvertibleNetworks, PyPlot, Flux, CUDA, JLD, GlowInvertibleNetwork
import Flux.Optimise: Optimiser, ADAM, ExpDecay, update!
using Random; Random.seed!(1)

# Create multiscale network
nc_hidden = 256
depth = 8
nscales = 3
cl_id = true
# cl_id = false
# conv_orth = true
conv_orth = false
conv_id = true
# conv_id = false
# cl_affine = true
cl_affine = false
G = Glow(1, nc_hidden, depth, nscales; logdet=false, conv_orth=conv_orth, cl_id=cl_id, conv_id=conv_id, cl_affine=cl_affine) |> gpu


# G = NetworkGlow(1, nc_hidden, nscales, depth) |> gpu
# G = NetworkGlow(1, nc_hidden, nscales, depth)


# Set loss function
function loss(X::AbstractArray{Float32,4}, Y::AbstractArray{Float32,4})
    ΔX = X-Y
    return 0.5f0*norm(ΔX)^2/size(X,4), ΔX/size(X,4)
end

# Setting optimizer options
batch_size = 2^1
nepochs = 200
lr = 1f-3
lrmin = lr*0.0001
opt = ADAM(lr)

# Training
floss = zeros(Float32, nepochs)
θ = get_params(G)

for e = 1:nepochs # epoch loop

    # Select mini-batch of data
    Xb = CUDA.randn(Float32, 64,64,1,batch_size)
    # Xb = randn(Float32, 64,64,1,batch_size)

    # Evaluate network
    Zb = G.forward(Xb)

    # Evaluate objective and gradients
    floss[e], ΔZb = loss(Zb, Xb)
    print("Iter: epoch=", e, "/", nepochs, "; f = ", floss[e], "\n")

    # Computing gradient
    G.backward(ΔZb, Zb)

    # Update params
    for p in θ
        update!(opt, p.data, p.grad)
        # p.grad = nothing # clear gradient
    end

end # end epoch loop