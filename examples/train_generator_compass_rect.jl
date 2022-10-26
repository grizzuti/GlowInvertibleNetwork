#CUDA_VISIBLE_DEVICES=6 nohup julia --project=. examples/train_generator_compass_rect.jl &
#################################################################################
# Train generator on MRI images
#################################################################################

using LinearAlgebra, InvertibleNetworks, PyPlot, Flux, CUDA,GlowInvertibleNetwork, Random
using MLUtils
import Flux.Optimise: Optimiser, ADAM, ExpDecay, ClipNorm, update!


using DrWatson

# Load data

using JLD2

batch_size = 2^4

function get_batch(X_3d, inds)
    batchsize = length(inds)
    x = zeros(Float32,nx,ny,1,batchsize)
    for i in 1:batchsize
        slice = X_3d[inds[i]...]'
        slice = (slice .- minimum(slice)) ./ (maximum(slice) .- minimum(slice))
        x[:,:,1,i] = slice
    end
    x
end

function get_data(; nx=128, ny=256)
    data_path = "../NormalizingFlow3D.jl/data/compass_volume.jld2"
    # if isfile(data_path) == false
    # println("Downloading data...");
    # download("https://www.dropbox.com/s/qesanry1w49rcju/compass_volume.jld2?dl=0", data_path)
    # end

    X_3d  = JLD2.jldopen(data_path, "r")["X"]

    n = size(X_3d)
    n_ava = n .- (ny+1,ny+1,nx+1)

    n_shift = 25
    @time inds_1 = [(i,  j:(j+ny-1),k:(k+nx-1)) for i in 1:n_shift:n[1] for j in 1:n_shift:n_ava[2] for k in 1:n_shift:n_ava[3]];
    @time inds_2 = [(i:(i+ny-1), j, k:(k+nx-1)) for i in 1:n_shift:n_ava[1] for j in 1:n_shift:n[2] for k in 1:n_shift:n_ava[3]];
    @time inds = vcat(inds_1,inds_2);

    n_total = length(inds)

    validation_perc = 0.95
    Random.seed!(123);
    inds_train, inds_test = splitobs(shuffle(inds); at=validation_perc, shuffle=true);

    num_plot = 8
    inds_plot = inds_train[1:num_plot]
    x = get_batch(X_3d, inds_plot)

    fig4 = figure(figsize=(10,5))
    for i in 1:8
        subplot(2,4,i); imshow(x[:,:,1,i]|> cpu);
    end
    tight_layout()
    fig_name = @strdict 
    safesave(joinpath(save_folder,savename(fig_name; digits=6)*"_train.png"), fig4); close(fig4)

    return X_3d, inds_train, inds_test
end

nx,ny = 256,512
N = nx*ny;


# Setting/loading intermediate/final saves
save_folder = "./results/compassgen/"
save_intermediate_filename = string(save_folder, "results_gen_intermediate_",nx,"x",ny,".jld")
save_filename = string(save_folder, "results_gen_",nx,"x",ny,".jld")


X_3d, inds_train, inds_test = get_data(;nx=nx,ny=ny)
n_train_curr = length(inds_train)
nbatches    = cld(n_train_curr,batch_size)-1
ntrain = nbatches*batch_size
inds_train = inds_train[1:ntrain]

n_test_curr = length(inds_test)
nbatches_test    = cld(n_test_curr,batch_size)-1
ntest = nbatches_test*batch_size
inds_test = inds_test[1:ntest]


println("ntrain=$(ntrain)")
println("ntest=$(ntest)")

# Create multiscale network
opts = GlowOptions(; cl_activation=SigmoidNewLayer(0.5f0),
                    cl_affine=true,
                    init_cl_id=true,
                    conv1x1_nvp=false,
                    init_conv1x1_permutation=true,
                    conv1x1_orth_fixed=true,
                    T=Float32)
nc = 1
nc_hidden = 512
depth = 5
nscales = 6
G = Glow(nc, nc_hidden, depth, nscales; opt=opts) |> gpu

# Set loss function
loss(X::AbstractArray{T,4}) where T = T(0.5)*norm(X)^2/size(X,4), X/size(X,4)

# Artificial data noise parameters
α = 0.1f0
β = 0.5f0
αmin = 0.01f0

# Setting optimizer options
#nbatches = Int64(ntrain/batch_size)
nepochs = 2^7
lr = 8f-5
clip_norm = 10f0
lr_min = lr*1f-2
decay_rate = exp(log(lr_min/lr)/(nepochs*nbatches))
λ = 1f-3

opt = Optimiser(ExpDecay(lr, decay_rate, 1, lr_min), ClipNorm(clip_norm),ADAM(lr))
intermediate_save = 1
intermediate_plot = 1

# Training
inds_b = inds_train[randperm(ntrain)[1:batch_size]]
X_init = get_batch(X_3d,inds_b)
G.forward(gpu(X_init)) # to initialize actnorm
θ = get_params(G);
θ_backup = deepcopy(θ);
floss = zeros(Float32, nbatches, nepochs)
floss_test = zeros(Float32, nbatches_test, nepochs)

floss_full = zeros(Float32, nbatches, nepochs)
floss_full_test = zeros(Float32,nbatches_test, nepochs)
Ztest = randn(Float32, nx,ny,1,batch_size);

for e = 1:nepochs # epoch loop
    # Select random data traversal for current epoch
    idx_e = reshape(randperm(ntrain), batch_size, nbatches)

    # Epoch-adaptive regularization weight
    λ_adaptive = λ*nx*ny/norm(θ_backup)^2

    @time begin
    for b = 1:nbatches # batch loop

        # Select mini-batch of data
        inds_b = inds_train[idx_e[:,b]]
        Xb = get_batch(X_3d,inds_b) |> gpu

        # Artificial noise
        noise_lvl = α/((e-1)*nbatches+b)^β+αmin
        println(noise_lvl)
        Xb .+= noise_lvl*CUDA.randn(Float32, size(Xb))

        # Evaluate network
        Zb, lgdt = G.forward(Xb)

        # Evaluate objective and gradients
        floss[b,e], ΔZb = loss(Zb)
        floss[b,e] /= N
        floss_full[b,e] = floss[b,e]-lgdt/N

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
    end

    idx_e_test = reshape(randperm(ntest), batch_size, nbatches_test)
    #test loop
    for b = 1:nbatches_test # batch loop
        # Select mini-batch of data
        inds_b = inds_test[idx_e_test[:,b]]
        Xb = get_batch(X_3d,inds_b) |> gpu

        # Artificial noise
        noise_lvl_test = 0.01#α/((e-1)*nbatches_test+b)^β+αmin
        println(noise_lvl_test)
        Xb .+= noise_lvl_test*CUDA.randn(Float32, size(Xb))

        # Evaluate network
        Zb, lgdt = G.forward(Xb)

        # Evaluate objective and gradients
        floss_test[b,e], ΔZb = loss(Zb)
    end # end batch loop

    #save_intermediate_filename = string(save_folder, "results_gen_intermediate_",nx,"x",e,".jld")
    # Saving and plotting intermediate results

    #JLD2.save(save_intermediate_filename, "theta", cpu(θ), "floss_full", floss_full, "floss", floss)
    if mod(e, intermediate_save ) == 0
         Params = get_params(G) |> cpu 
         save_dict = @strdict nx ny ntrain e Params floss_full floss nc_hidden depth nscales clip_norm
         @tagsave(
             "data/"*savename(save_dict, "jld2"; digits=6),
             save_dict;
             safe=true
         )
    end
    if mod(e, intermediate_plot) == 0
        X_test = G.inverse(gpu(Ztest)) |> cpu
        plot_image(X_test[:, :, 1, 1]; figsize=(5,5),  title=L"$\mathbf{x}$", path=string(save_folder, "new_samples_1intermediate_",nx,"x", e,"_b.png"))
        plot_image(X_test[:, :, 1, 2]; figsize=(5,5),  title=L"$\mathbf{x}$", path=string(save_folder, "new_samples_2intermediate_",nx,"x", e,"_b.png"))
        plot_image(X_test[:, :, 1, 3]; figsize=(5,5), title=L"$\mathbf{x}$", path=string(save_folder, "new_samples_3intermediate_",nx,"x", e,"_b.png"))
        plot_image(X_test[:, :, 1, 4]; figsize=(5,5), title=L"$\mathbf{x}$", path=string(save_folder, "new_samples_4intermediate_",nx,"x", e,"_b.png"))
        
        plot_loss(range(0, e, length=length(floss_full[:,1:e])), vec(floss_full[:,1:e]); figsize=(7, 2.5), color="#d48955", title="Negative log-likelihood", path=string(save_folder, "loss_intermediate_",nx,"x", e,".png"), xlabel="Epochs", ylabel="Training objective")
        plot_loss(range(0, e, length=length(floss[:,1:e])), vec(floss[:,1:e]); figsize=(7, 2.5), color="#d48955", title="Negative log-likelihood", path=string(save_folder, "loss_intermediate_z_",nx,"x", e,".png"), xlabel="Epochs", ylabel="Training objective")
       
        plot_loss(range(0, e, length=length(floss_test[:,1:e])), vec(floss_test[:,1:e]); figsize=(7, 2.5), color="#d48955", title="Negative log-likelihood", path=string(save_folder, "test_loss_intermediate_z_",nx,"x", e,".png"), xlabel="Epochs", ylabel="Training objective")
    end

end # end epoch loop

# Final save
save(save_filename, "theta", cpu(get_params(G)), "floss_full", floss_full, "floss", floss)