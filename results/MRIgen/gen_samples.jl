using GlowInvertibleNetwork, LinearAlgebra, InvertibleNetworks, PyPlot, JLD
using Random; Random.seed!(1)

# Load network weights
# θtrained = load("./results/MRIgen/results_gen_300.jld")["theta"]
θtrained = load("./results/MRIgen/results_gen_1000.jld")["theta"]

# Load data
X = Float32.(reshape(load("./data/AOMIC_data64x64.jld")["data"], 64, 64, 1, :))
ntrain = size(X, 4)

# Create multiscale network
nc_hidden = 512
depth = 5
nscales = 5
cl_id = true
# cl_id = false
conv_orth = true
# conv_orth = false
conv_id = true
# conv_id = false
cl_activation = SigmoidNewLayer(0.5f0)
G = Glow(1, nc_hidden, depth, nscales; conv_orth=conv_orth, cl_id=cl_id, cl_activation=cl_activation, conv_id=conv_id)

# Set weights
set_params!(G, θtrained)
G.forward(randn(Float32, 64,64,1,16))

# Generate new samples
nsamples = 2^4
Z = randn(Float32, 64,64,1,nsamples)
Xnew = G.inverse(Z)

# Plotting
fig1 = figure()
n = 4
for i = 1:n
    subplot(1,n,i)
    imshow(X[:,:,1,i], aspect=1, resample=true, interpolation="none", filterrad=1, cmap="gray")
    axis("off")
    title(L"$x\sim p(x)$")
end
savefig("./results/MRIgen/true_samples.png", format="png", bbox_inches="tight", dpi=300, pad_inches=.05)
close(fig1)

fig2 = figure()
n = 4
for i = 1:n
    subplot(1,n,i)
    imshow(Xnew[:,:,1,i], aspect=1, resample=true, interpolation="none", filterrad=1, cmap="gray")
    axis("off")
    title(L"$x\sim p_{\theta}(x)$")
end
savefig("./results/MRIgen/new_samples.png", format="png", bbox_inches="tight", dpi=300, pad_inches=.05)
close(fig2)