using GlowInvertibleNetwork, LinearAlgebra, InvertibleNetworks, PyPlot, JLD, Distributions, Statistics
using Random; Random.seed!(1)
include("../../examples/plotting_utils.jl")

# Load network weights
θtrained = load("./results/MRIgen/results_gen_intermediate_256x256.jld")["theta"]

# Load loss functions
f_loss = load("./results/MRIgen/results_gen_intermediate_256x256.jld")["floss"]
f_loss_logdet = load("./results/MRIgen/results_gen_intermediate_256x256.jld")["floss_logdet"]

# Load data
X = Float32.(reshape(load("./data/AOMIC_data256x256.jld")["data"], 256,256,1,:))
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
G = Glow(nc, nc_hidden, depth, nscales; opt=opt)

# Set weights
set_params!(G, θtrained)
G.forward(randn(Float32, 256,256,1,16))

# Generate new samples
nsamples = 2^4
Znew = randn(Float32, 256,256,1,nsamples)
Xnew = G.inverse(Znew)
Znew_hi = 0.9f0*randn(Float32, 256,256,1,nsamples)
Xnew_hi = G.inverse(Znew_hi)

# Generate latent-space codes
nsamples = 2^4
αmin = 0.01f0
Xsamples = X[:,:,:,randperm(size(X,4))[1:nsamples]]+αmin*randn(Float32,256,256,1,nsamples)
Zsamples,_ = G.forward(Xsamples)

# Plotting
fig = figure()
n = 4
F⁰ = Normal(0,1)
for i = 1:n
    subplot(3,n,i)
    imshow(Xsamples[:,:,1,i], aspect=1, resample=true, interpolation="none", filterrad=1, cmap="gray", vmin=0, vmax=1)
    axis("off")
    title(L"$x\sim p(x)$")
    subplot(3,n,i+n)
    imshow(Zsamples[:,:,1,i], aspect=1, resample=true, interpolation="none", filterrad=1, cmap="gray", vmin=-3, vmax=3)
    axis("off")
    title(L"$z\sim p_{\theta}(z)$")
    subplot(3,n,i+2*n)
    obs = vec(Zsamples[:,:,1,i])
    nobs = length(obs); sort!(obs)
    quantiles⁰ = [quantile(F⁰, i/nobs) for i in 1:nobs]
    PyPlot.scatter(quantiles⁰, obs, s=0.5)
    PyPlot.plot(obs,obs,color="red",label="")
    title("qq plot"); xlim(-5,5); ylim(-5,5);
    xlabel("Theoretical q."); i==1 && ylabel("Sample q.")
end
savefig("./results/MRIgen/true_samples_256x256.png", format="png", bbox_inches="tight", dpi=300, pad_inches=.05)
close(fig)

fig = figure()
n = 4
for i = 1:n
    subplot(1,n,i)
    imshow(Xnew[:,:,1,i], aspect=1, resample=true, interpolation="none", filterrad=1, cmap="gray", vmin=0, vmax=1)
    axis("off")
    title(L"$x\sim p_{\theta}(x)$")
end
savefig("./results/MRIgen/new_samples_256x256.png", format="png", bbox_inches="tight", dpi=300, pad_inches=.05)
close(fig)

fig = figure()
n = 4
for i = 1:n
    subplot(1,n,i)
    imshow(Xnew_hi[:,:,1,i], aspect=1, resample=true, interpolation="none", filterrad=1, cmap="gray", vmin=0, vmax=1)
    axis("off")
    title(L"$x\sim p_{\theta}(x)$")
end
savefig("./results/MRIgen/new_samples_hi_256x256.png", format="png", bbox_inches="tight", dpi=300, pad_inches=.05)
close(fig)

nepochs = size(f_loss,2)
plot_loss(range(0, nepochs, length=length(f_loss_logdet[:])), vec(f_loss_logdet); figsize=(7, 2.5), color="#d48955", title=L"$\mathbb{E}_{\mathbf{x}\sim p_X(\mathbf{x})}\ \frac{1}{2}||f_{\theta}(\mathbf{x})||^2-\log|J_{f_{\theta}}(\mathbf{x})|$", path="results/MRIgen/loss_full_256x256.png", xlabel="Epochs", ylabel="Training objective")

const_loss = 0.5f0*prod(size(X)[1:2])*ones(Float32,length(f_loss),1)
fig = figure(figsize=(7, 2.5))
n = range(0, nepochs, length=length(f_loss[:]))
plot(n, vec(f_loss), color="#d48955")
plot(n, const_loss, "r--")
PyPlot.title(L"$\mathbb{E}_{\mathbf{x}\sim p_X(\mathbf{x})}\ \frac{1}{2}||f_{\theta}(\mathbf{x})||^2$")
PyPlot.ylabel("Training objective")
PyPlot.xlabel("Epochs")
grid(true, which="major")
grid(true, which="minor")
savefig("results/MRIgen/loss_z_256x256.png", format="png", bbox_inches="tight", dpi=300, pad_inches=.05)
close(fig)