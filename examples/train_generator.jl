#################################################################################
# Train generator on MRI images
#################################################################################

using LinearAlgebra, InvertibleNetworks, PyPlot, Flux, CUDA, JLD, GlowInvertibleNetwork, Random
import Flux.Optimise: Optimiser, ADAM, ExpDecay, ClipNorm, update!
include("./plotting_utils.jl")


# Load data
nx,ny = 256,256
data_filename = string("./data/AOMIC_data",nx,"x",ny,".jld")
X = reshape(Float32.(load(data_filename)["data"]), nx,ny,1,:)
ntrain = size(X,4)

# Setting/loading intermediate/final saves
save_folder = "./results/MRIgen/"
save_intermediate_filename = string(save_folder, "results_gen_intermediate_",nx,"x",ny,".jld")
save_filename = string(save_folder, "results_gen_",nx,"x",ny,".jld")

# Create multiscale network
opt = GlowOptions(; cl_activation=SigmoidNewLayer(0.5f0),
                    cl_affine=true,
                    init_cl_id=true,
                    conv1x1_nvp=false,
                    init_conv1x1_permutation=true,
                    conv1x1_orth_fixed=true,
                    T=Float32)
nc = 1
nc_hidden = 512
depth = 5
nscales = 7
G = Glow(nc, nc_hidden, depth, nscales; opt=opt) |> gpu

# Set loss function
loss(X::AbstractArray{T,4}) where T = T(0.5)*norm(X)^2/size(X,4), X/size(X,4)

# Artificial data noise parameters
α = 0.1f0
β = 0.5f0
αmin = 0.01f0

# Setting optimizer options
batch_size = 2^2
nbatches = Int64(ntrain/batch_size)
nepochs = 2^9
lr = 1f-4
lr_min = lr*1f-2
decay_rate = exp(log(lr_min/lr)/(nepochs*nbatches))
grad_max = lr*1f7
opt = Optimiser(ClipNorm(grad_max), ExpDecay(lr, decay_rate, 1, lr_min), ADAM(lr))
intermediate_save = 1

# Training
floss = zeros(Float32, nbatches, nepochs)
floss_full = zeros(Float32, nbatches, nepochs)
Ztest = randn(Float32, nx,ny,1,batch_size)
for e = 1:nepochs # epoch loop

    # Set backup state
    θ_backup = get_params(G) |> cpu

    # Select random data traversal for current epoch
    idx_e = reshape(randperm(ntrain), batch_size, nbatches)

    for b = 1:nbatches # batch loop

        # Select mini-batch of data
        Xb = X[:,:,:,idx_e[:,b]] |> gpu

        # Artificial noise
        noise_lvl = α/((e-1)*nbatches+b)^β+αmin
        Xb .+= noise_lvl*CUDA.randn(Float32, size(Xb))

        # Evaluate network
        Zb, lgdt = G.forward(Xb)

        # Evaluate objective and gradients
        floss[b,e], ΔZb = loss(Zb)
        floss_full[b,e] = floss[b,e]-lgdt

        # Print current iteration
        print("Iter: epoch=", e, "/", nepochs, ", batch=", b, "/", nbatches, "; f_full = ", floss_full[b,e], "; f_z = ", floss[b,e], "\n")

        # Check instability status
        if isnan(floss_full[b,e]) || isinf(floss_full[b,e])
            set_params!(G, gpu(θ_backup))
            save(save_intermediate_filename, "theta", cpu(get_params(G)), "floss_full", floss_full, "floss", floss)
            throw("NaN or Inf values!\n")
        end

        # Computing gradient
        G.backward(ΔZb, Zb)

        # Update params
        for p in get_params(G)
            update!(opt, p.data, p.grad)
            p.grad = nothing # clear gradient
        end

    end # end batch loop

    # Saving and plotting intermediate results
    save(save_intermediate_filename, "theta", cpu(get_params(G)), "floss_full", floss_full, "floss", floss)
    if mod(e, intermediate_save) == 0
        X_test = G.inverse(gpu(Ztest)) |> cpu
        plot_image(X_test[:, :, 1, 1]; figsize=(5,5), vmin=min(X_test[:,:,1,1]...), vmax=max(X_test[:,:,1,1]...), title=L"$\mathbf{x}$", path=string(save_folder, "new_samples_intermediate_",nx,"x", ny,".png"))
        plot_loss(range(0, e, length=length(floss_full[:,1:e])), vec(floss_full[:,1:e]); figsize=(7, 2.5), color="#d48955", title="Negative log-likelihood", path=string(save_folder, "loss_intermediate_",nx,"x", ny,".png"), xlabel="Epochs", ylabel="Training objective")
        plot_loss(range(0, e, length=length(floss[:,1:e])), vec(floss[:,1:e]); figsize=(7, 2.5), color="#d48955", title="Negative log-likelihood", path=string(save_folder, "loss_intermediate_z_",nx,"x", ny,".png"), xlabel="Epochs", ylabel="Training objective")
    end

end # end epoch loop

# Final save
save(save_filename, "theta", cpu(get_params(G)), "floss_full", floss_full, "floss", floss)