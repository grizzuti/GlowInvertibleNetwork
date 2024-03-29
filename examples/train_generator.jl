#################################################################################
# Train generator on MRI images
#################################################################################

using LinearAlgebra, InvertibleNetworks, PyPlot, Flux, ParameterSchedulers, CUDA, JLD, GlowInvertibleNetwork, Random
import Flux.Optimise: Optimiser, ADAM, ExpDecay, ClipNorm, update!
import ParameterSchedulers: Scheduler
include("./plotting_utils.jl")


# Load data
nx, ny = 64, 64
# nx, ny = 256, 256
data_filename = string("./data/AOMIC_data", nx, "x", ny, ".jld")
X = reshape(Float32.(load(data_filename)["data"]), nx, ny, 1, :)
ntrain = size(X, 4)

# Setting/loading intermediate/final saves
save_folder = "./results/MRIgen/"
save_intermediate_filename = string(save_folder, "results_gen_intermediate_", nx, "x", ny, ".jld")
save_filename = string(save_folder, "results_gen_", nx, "x", ny, ".jld")

# Create multiscale network
nc = 1
nc_hidden = 512
depth = 5
scales = 6
G = Glow(nc; nc_hidden=nc_hidden, init_id_cl=true, depth, scales, logdet=true, initial_squeeze=true, ndims=2) |> gpu

# Set loss function
loss(X) = norm(X)^2/(2*size(X,4)), X/size(X,4)

# Artificial data noise parameters
σ_noise = 1f-3

# Setting optimizer options
batch_size = 2^2
nbatches = Int(ntrain/batch_size)
nepochs = 2^9
lr = 1f-4
lr_sched = CosAnneal(; λ0=lr, λ1=1f-2*lr, period=nepochs*nbatches)
opt = Scheduler(lr_sched, ADAM(lr))
intermediate_save = 1

# Training
G.forward(gpu(X[:, :, :, randperm(ntrain)[1:batch_size]])) # to initialize actnorm
θ = get_params(G)
θ_backup = deepcopy(θ)
floss = zeros(Float32, nbatches, nepochs)
floss_full = zeros(Float32, nbatches, nepochs)
Ztest = randn(Float32, nx, ny, 1, batch_size)
for e = 1:nepochs # epoch loop

    # Select random data traversal for current epoch
    idx_e = reshape(randperm(ntrain), batch_size, nbatches)

    for b = 1:nbatches # batch loop

        # Select mini-batch of data
        Xb = X[:,:,:,idx_e[:,b]] |> gpu

        # Artificial noise
        Xb .+= σ_noise*CUDA.randn(Float32, size(Xb))

        # Evaluate network
        Zb, lgdt = G.forward(Xb)

        # Evaluate objective and gradients
        floss[b,e], ΔZb = loss(Zb)
        floss_full[b,e] = floss[b,e]-lgdt

        # Computing gradient
        G.backward(ΔZb, Zb)

        # Update params + regularization
        for i = eachindex(θ)
            update!(opt, θ[i].data, θ[i].grad)
        end

        # Print current iteration
        print("Iter: epoch=", e, "/", nepochs, ", batch=", b, "/", nbatches, "; f_full = ", floss_full[b,e], "; f_z = ", floss[b,e], "\n")

    end # end batch loop

    # Saving and plotting intermediate results
    save(save_intermediate_filename, "theta", cpu(θ), "floss_full", floss_full, "floss", floss)
    if mod(e, intermediate_save) == 0
        X_test = G.inverse(gpu(Ztest)) |> cpu
        plot_image(X_test[:, :, 1, 1]; figsize=(5,5), vmin=min(X_test[:,:,1,1]...), vmax=max(X_test[:,:,1,1]...), title=L"$\mathbf{x}$", path=string(save_folder, "new_samples_intermediate_",nx,"x", ny,".png"))
        plot_loss(range(0, e, length=length(floss_full[:,1:e])), vec(floss_full[:,1:e]); figsize=(7, 2.5), color="#d48955", title="Negative log-likelihood", path=string(save_folder, "loss_intermediate_",nx,"x", ny,".png"), xlabel="Epochs", ylabel="Training objective")
        plot_loss(range(0, e, length=length(floss[:,1:e])), vec(floss[:,1:e]); figsize=(7, 2.5), color="#d48955", title="Negative log-likelihood", path=string(save_folder, "loss_intermediate_z_",nx,"x", ny,".png"), xlabel="Epochs", ylabel="Training objective")
    end

end # end epoch loop

# Final save
save(save_filename, "theta", cpu(get_params(G)), "floss_full", floss_full, "floss", floss)