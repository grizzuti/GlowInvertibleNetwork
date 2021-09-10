#################################################################################
# Train generator on MRI images
#################################################################################

using LinearAlgebra, InvertibleNetworks, PyPlot, Flux, CUDA, JLD, GlowInvertibleNetwork
import Flux.Optimise: Optimiser, ADAM, ExpDecay, update!
using Random; Random.seed!(1)
include("./plotting_utils.jl")


# Load data
X = reshape(load("./data/AOMIC_data64x64.jld")["data"], 64, 64, 1, :) |> gpu
ntrain = size(X, 4)

# Create multiscale network
nc_hidden = 512
depth = 5
nscales = 5
cl_id = true
# cl_id = false
# conv_orth = true
conv_orth = false
conv_id = true
# conv_id = false
cl_affine = true
# cl_affine = false
G = Glow(1, nc_hidden, depth, nscales; conv_orth=conv_orth, cl_id=cl_id, conv_id=conv_id, cl_affine=cl_affine) |> gpu

# Set loss function
loss(X::AbstractArray{Float32,4}) = 0.5f0*norm(X)^2/size(X,4), X/size(X,4)

# Setting optimizer options
batch_size = 2^4
nbatches = Int64(ntrain/batch_size)
nepochs = 10
lr = 1f-4
lrmin = lr*0.0001
decay_rate = 0.3
lr_step = Int64(floor(nepochs*nbatches*log(decay_rate)/log(lrmin/lr)))
opt = Optimiser(ExpDecay(lr, decay_rate, lr_step, lrmin), ADAM(lr))
grad_clip = false; grad_max = 5f0

# Training
floss = zeros(Float32, nbatches, nepochs)
floss_logdet = zeros(Float32, nbatches, nepochs)
θ = get_params(G)

for e = 1:nepochs # epoch loop

    # Select random data traversal for current epoch
    idx_e = reshape(randperm(ntrain), batch_size, nbatches)

    for b = 1:nbatches # batch loop

        # Select mini-batch of data
        Xb = X[:,:,:,idx_e[:,b]]
        Xb .+= 0.01f0*CUDA.randn(Float32, size(Xb))

        # Evaluate network
        Zb, lgdet = G.forward(Xb)

        # Evaluate objective and gradients
        floss[b,e], ΔZb = loss(Zb); floss_logdet[b,e] = floss[b,e]-lgdet
        print("Iter: epoch=", e, "/", nepochs, ", batch=", b, "/", nbatches, "; f = ", floss_logdet[b,e], "\n")

        # Computing gradient
        G.backward(ΔZb, Zb)

        # Update params
        for p in θ
            grad_clip && (norm(p.grad) > grad_max) && (p.grad *= grad_max/norm(p.grad)) # Gradient clipping
            update!(opt, p.data, p.grad)
            # p.grad = nothing # clear gradient
        end

    end # end batch loop

    # Saving intermediate results
    (mod(e, 10)==0 || e==nepochs) && save("./results/MRIgen/results_gen.jld", "theta", cpu(θ), "floss_logdet", floss_logdet, "floss", floss)

end # end epoch loop

# Generate random samples
G = G |> cpu
Z = randn(Float32,64,64,1,nbatches)
X_new = G.inverse(Z)

# Plotting
plot_image(X_new[:, :, 1, 1]; figsize=(5,5), vmin=min(X_new[:,:,1,1]...), vmax=max(X_new[:,:,1,1]...), title=L"$\mathbf{x}$", path="results/MRIgen/new_samples.png")
plot_loss(range(0, nepochs, length=length(floss_logdet[:])), vec(floss_logdet); figsize=(7, 2.5), color="#d48955", title="Negative log-likelihood", path="results/MRIgen/loss.png", xlabel="Epochs", ylabel="Training objective")
save("./results/MRIgen/results_gen.jld", "theta", cpu(θ), "floss_logdet", floss_logdet, "floss", floss, "samples", X_new)
