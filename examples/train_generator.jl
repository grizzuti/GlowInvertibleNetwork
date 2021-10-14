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
opt = GlowOptions(; cl_activation=SigmoidNewLayer(0.05f0),
                    cl_affine=true,
                    init_cl_id=true,
                    conv1x1_nvp=false,
                    init_conv1x1_permutation=true,
                    conv1x1_orth_fixed=true,
                    T=Float32)
nc = 1
nc_hidden = 512
depth = 5
nscales = 5
G = Glow(nc, nc_hidden, depth, nscales; opt=opt) |> gpu

# Set loss function
loss(X::AbstractArray{Float32,4}) = 0.5f0*norm(X)^2/size(X,4), X/size(X,4)

# Setting optimizer options
batch_size = 2^4
nbatches = Int64(ntrain/batch_size)
nepochs = 1000
lr = 1f-4
# lrmin = lr*0.0001
lrmin = lr*0.99
decay_rate = 0.3
lr_step = Int64(floor(nepochs*nbatches*log(decay_rate)/log(lrmin/lr)))
opt = Optimiser(ExpDecay(lr, decay_rate, lr_step, lrmin), ADAM(lr))
grad_clip = false; grad_max = 0.0001f0
intermediate_save = 10

# Test latent
ntest = batch_size
Ztest = randn(Float32, 64, 64, 1, ntest) |> gpu

# Training
floss = zeros(Float32, nbatches, nepochs)
floss_logdet = zeros(Float32, nbatches, nepochs)
θ = get_params(G)

global α = nothing # calibration factor
for e = 1:nepochs # epoch loop

    # Select random data traversal for current epoch
    idx_e = reshape(randperm(ntrain), batch_size, nbatches)

    for b = 1:nbatches # batch loop

        # Select mini-batch of data
        Xb = X[:,:,:,idx_e[:,b]]

        # Evaluate network
        Zb, lgdt = G.forward(Xb)
        # (α === nothing) && (global α = sqrt(2f0*prod(size(Zb)[1:2])/(2f0*loss(Zb)[1])))
        # (b == 1) && (global α = sqrt(prod(size(Zb)[1:2])/(2f0*loss(Zb)[1])))
        # global α = sqrt(prod(size(Zb)[1:2])/(2f0*loss(Zb)[1]))
        # Zb *= α

        # Evaluate objective and gradients
        floss[b,e], ΔZb = loss(Zb); floss_logdet[b,e] = floss[b,e]-lgdt
        print("Iter: epoch=", e, "/", nepochs, ", batch=", b, "/", nbatches, "; f_full = ", floss_logdet[b,e], "; f_z = ", floss[b,e], "\n")

        # Reload previous state to prevent divergence
        if e > 1 && (isnan(floss_logdet[b,e]) || isinf(floss_logdet[b,e]))
            print("Ooops!\n")
            set_params!(G, gpu(load("./results/MRIgen/results_gen_intermediate.jld")["theta"]))
            break
        end

        # Computing gradient
        G.backward(ΔZb, Zb)
        # G.backward(α.*ΔZb, Zb)

        # Update params
        for p in θ
            grad_clip && (norm(p.grad) > grad_max) && (p.grad *= grad_max/norm(p.grad)) # Gradient clipping
            update!(opt, p.data, p.grad)
            p.grad = nothing # clear gradient
        end

    end # end batch loop

    # Saving intermediate results
    save("./results/MRIgen/results_gen_intermediate.jld", "theta", cpu(θ), "floss_logdet", floss_logdet, "floss", floss)
    if mod(e, intermediate_save) == 0
        X_test = G.inverse(Ztest) |> cpu
        plot_image(X_test[:, :, 1, 1]; figsize=(5,5), vmin=min(X_test[:,:,1,1]...), vmax=max(X_test[:,:,1,1]...), title=L"$\mathbf{x}$", path="results/MRIgen/new_samples_intermediate.png")
        plot_loss(range(0, nepochs, length=length(floss_logdet[:])), vec(floss_logdet); figsize=(7, 2.5), color="#d48955", title="Negative log-likelihood", path="results/MRIgen/loss_intermediate.png", xlabel="Epochs", ylabel="Training objective")
        plot_loss(range(0, nepochs, length=length(floss[:])), vec(floss); figsize=(7, 2.5), color="#d48955", title="Negative log-likelihood", path="results/MRIgen/loss_intermediate_z.png", xlabel="Epochs", ylabel="Training objective")
    end

end # end epoch loop

# Generate random samples
G = G |> cpu
Z = randn(Float32,64,64,1,batch_size)
X_new = G.inverse(Z)

# Plotting
plot_image(X_new[:, :, 1, 1]; figsize=(5,5), vmin=min(X_new[:,:,1,1]...), vmax=max(X_new[:,:,1,1]...), title=L"$\mathbf{x}$", path="results/MRIgen/new_samples.png")
plot_loss(range(0, nepochs, length=length(floss_logdet[:])), vec(floss_logdet); figsize=(7, 2.5), color="#d48955", title="Negative log-likelihood", path="results/MRIgen/loss.png", xlabel="Epochs", ylabel="Training objective")
plot_loss(range(0, nepochs, length=length(floss[:])), vec(floss); figsize=(7, 2.5), color="#d48955", title="Negative log-likelihood", path="results/MRIgen/loss_z.png", xlabel="Epochs", ylabel="Training objective")
save("./results/MRIgen/results_gen.jld", "theta", cpu(θ), "floss_logdet", floss_logdet, "floss", floss, "samples", X_new)