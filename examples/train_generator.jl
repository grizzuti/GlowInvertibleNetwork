#CUDA_VISIBLE_DEVICES=1 nohup julia --project=. examples/train_generator.jl &
#################################################################################
# Train generator on MRI images
#################################################################################

using LinearAlgebra, InvertibleNetworks, PyPlot, Flux, CUDA, JLD, GlowInvertibleNetwork, Random
import Flux.Optimise: Optimiser, ADAM, ExpDecay, ClipNorm, update!
include("plotting_utils.jl")

using DrWatson
#using JLD2
#nx,ny = 256,256
#data_filename = string("data/AOMIC_data",nx,"x",ny,".jld")
#X = JLD2.load(data_filename)["data"]

# Load data
nx,ny = 256,256
#data_filename = string("data/AOMIC_data",nx,"x",ny,".jld")
#X = reshape(Float32.(load(data_filename)["data"]), nx, ny,1,:)

using JLD2

#JLD2.@save "AOMIC_data_256.jld2" X
#JLD2.@load "AOMIC_data_256.jld2" X
JLD2.@load "../MultiSourceSummary.jl/normalized_train_256_norm01.jld2" Xs

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
λ = 1f-3
opt = Optimiser(ExpDecay(lr, decay_rate, 1, lr_min), ADAM(lr))
intermediate_save = 1
intermediate_plot = 1

# Training
G.forward(gpu(X[:,:,:,randperm(ntrain)[1:batch_size]])) # to initialize actnorm
θ = get_params(G);
θ_backup = deepcopy(θ);
floss = zeros(Float32, nbatches, nepochs)
floss_full = zeros(Float32, nbatches, nepochs)
Ztest = randn(Float32, nx,ny,1,batch_size)
for e = 1:nepochs # epoch loop

    # Select random data traversal for current epoch
    idx_e = reshape(randperm(ntrain), batch_size, nbatches)

    # Epoch-adaptive regularization weight
    λ_adaptive = λ*nx*ny/norm(θ_backup)^2

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

        # Computing gradient
        G.backward(ΔZb, Zb)

        # Update params + regularization
        for i =1:length(θ)
            Δθ = θ[i].data-θ_backup[i].data
            update!(opt, θ[i].data, θ[i].grad+λ_adaptive*Δθ)
            floss_full[b,e] += 0.5f0*λ_adaptive*norm(Δθ)^2
            (b == nbatches) && (θ_backup[i].data .= θ[i].data)
        end

        # Print current iteration
        print("Iter: epoch=", e, "/", nepochs, ", batch=", b, "/", nbatches, "; f_full = ", floss_full[b,e], "; f_z = ", floss[b,e], "\n")
        Base.flush(Base.stdout)

        # Check instability status
        if isnan(floss_full[b,e]) || isinf(floss_full[b,e])
            set_params!(G, θ_backup)
            save(save_intermediate_filename, "theta", cpu(θ), "floss_full", floss_full, "floss", floss)
            throw("NaN or Inf values!\n")
        end

    end # end batch loop

    #save_intermediate_filename = string(save_folder, "results_gen_intermediate_",nx,"x",e,".jld")
    # Saving and plotting intermediate results

    #JLD2.save(save_intermediate_filename, "theta", cpu(θ), "floss_full", floss_full, "floss", floss)
    if mod(e, intermediate_save ) == 0
         Params = get_params(G) |> cpu 
         save_dict = @strdict e Params floss_full floss nc_hidden depth nscales 
         @tagsave(
             "data/"*savename(save_dict, "jld2"; digits=6),
             save_dict;
             safe=true
         )
    end
    if mod(e, intermediate_plot) == 0
        X_test = G.inverse(gpu(Ztest)) |> cpu
        plot_image(X_test[:, :, 1, 1]; figsize=(5,5),  title=L"$\mathbf{x}$", path=string(save_folder, "new_samples_1intermediate_",nx,"x", e,".png"))
        plot_image(X_test[:, :, 1, 2]; figsize=(5,5),  title=L"$\mathbf{x}$", path=string(save_folder, "new_samples_2intermediate_",nx,"x", e,".png"))
        plot_image(X_test[:, :, 1, 3]; figsize=(5,5), title=L"$\mathbf{x}$", path=string(save_folder, "new_samples_3intermediate_",nx,"x", e,".png"))
        plot_image(X_test[:, :, 1, 4]; figsize=(5,5), title=L"$\mathbf{x}$", path=string(save_folder, "new_samples_4intermediate_",nx,"x", e,".png"))
        
        plot_loss(range(0, e, length=length(floss_full[:,1:e])), vec(floss_full[:,1:e]); figsize=(7, 2.5), color="#d48955", title="Negative log-likelihood", path=string(save_folder, "loss_intermediate_",nx,"x", e,".png"), xlabel="Epochs", ylabel="Training objective")
        plot_loss(range(0, e, length=length(floss[:,1:e])), vec(floss[:,1:e]); figsize=(7, 2.5), color="#d48955", title="Negative log-likelihood", path=string(save_folder, "loss_intermediate_z_",nx,"x", e,".png"), xlabel="Epochs", ylabel="Training objective")
    end

end # end epoch loop

# Final save
save(save_filename, "theta", cpu(get_params(G)), "floss_full", floss_full, "floss", floss)