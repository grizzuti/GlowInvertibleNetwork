#export CUDA_VISIBLE_DEVICES=5
#CUDA_VISIBLE_DEVICES=5 nohup julia --project=.  examples/map_2d_deep_prior.jl &
#using DrWatson
#@quickactivate :NormalizingFlow3D
#import Pkg; Pkg.instantiate()

#CUDA_VISIBLE_DEVICES=5 julia --project=.

using LinearAlgebra, InvertibleNetworks, PyPlot, Flux, CUDA,GlowInvertibleNetwork, Random
using MLUtils
import Flux.Optimise: Optimiser, ADAM, ExpDecay, ClipNorm, update!

using DrWatson
using JUDI 
using JLD2
using Flux 
using Images
using PyPlot 
using Statistics 
using JOLI 
using ImageQualityIndexes
using CUDA
using SlimPlotting
using SlimOptim

font_size = 12
PyPlot.rc("font", family="serif", size=font_size); PyPlot.rc("xtick", labelsize=font_size); PyPlot.rc("ytick", labelsize=font_size);
PyPlot.rc("axes", labelsize=font_size)    # fontsize of the x and y labels

# Plotting path
experiment_name = "map_2d"
plot_path = "plots/map_2d"
data_path = "../NormalizingFlow3D.jl/data/compass_volume.jld2"

NewMin = 1.480f0
NewMax = 4.5f0
idx_wb = 21

function get_v(m)
    sqrt.(1f0./m)
end

function water_mute(v)
    return hcat(NewMin * ones(Float32, n[1], idx_wb), 0f0 * zeros(Float32, n[1], n[2]-idx_wb)) + hcat(0f0 * zeros(Float32, n[1], idx_wb), ones(Float32, n[1], n[2]-idx_wb)) .* v
end

function get_batch(X_3d, inds)
    batchsize = length(inds)
    x = zeros(Float32,nx,ny,1,batchsize)

    for i in 1:batchsize
        slice = X_3d[inds[i]...] 
        slice /= 1000f0 
        x[:,:,1,i] = slice
    end
    x
end

nx,ny = 512,256
X_3d  = JLD2.jldopen(data_path, "r")["X"]
inds_test = [(176:687, 851, 26:281)]
X_gt = get_batch(X_3d, inds_test)[:,:,1,1]

# Set up model structure
n = size(X_gt) # (x,y,z) or (x,z)
d = (25., 25.) # true d
o = (0., 0.)

# Velocity [km/s]
v  = X_gt

fig=figure(figsize=(15,7));
imshow(v[:,:]'); colorbar()

tight_layout()
fig_name = @strdict
safesave(joinpath(plot_path,savename(fig_name; digits=6)*"_v_get.png"), fig); close(fig)

v0 = imfilter(v, Kernel.gaussian(20f0))

v = water_mute(v)
v0 = water_mute(v0)

# Bound constraints
vmin = ones(Float32, n) .* NewMin#1.5f0
vmax = ones(Float32, n) .* NewMax#(shift + 1f0)#3.5f0

vmin[:,1:idx_wb] .= v0[:,1:idx_wb]   # keep water column fixed
vmax[:,1:idx_wb] .= v0[:,1:idx_wb]

# Slowness squared [s^2/km^2]
mmin = (1f0 ./ vmax).^2
mmax = (1f0 ./ vmin).^2

# Slowness squared [s^2/km^2]
m = (1f0 ./ v).^2
m0 = (1f0 ./ v0).^2

# Setup model structure
nsrc = 2    # number of sources
model = Model(n, d, o, m)

#' ## Create source and receivers positions at the surface
# Set up receiver geometry
nxrec = 512
xrec = range(d[1], stop=(n[1]-1)*d[1], length=nxrec)
yrec = 0f0 # WE have to set the y coordiante to zero (or any number) for 2D modeling
zrec = range(d[1], stop=d[1], length=nxrec)

# receiver sampling and recording time
timeD = 7000f0   # receiver recording time [ms] # try going down to 600
dtD = 4f0    # receiver sampling interval [ms]

data_extent = (0, nxrec, timeD, 0)
model_extent = (0,(n[1]-1)*d[1],(n[2]-1)*d[2],0)

# Set up receiver structure
recGeometry = Geometry(xrec, yrec, zrec; dt=dtD, t=timeD, nsrc=nsrc)

# Set up source structure
xsrc = convertToCell(range(d[1],stop=(n[1]-1)*d[1], length=nsrc))
ysrc = convertToCell(range(0f0,stop=0f0,length=nsrc))
zsrc = convertToCell(range((idx_wb-1)*d[2]-2f0,stop=(idx_wb-1)*d[2]-2f0,length=nsrc))

srcGeometry = Geometry(xsrc, ysrc, zsrc; dt=dtD, t=timeD)

# setup wavelet 
#f0 = 0.025f0     # kHz
#wavelet = ricker_wavelet(timeD, dtD, f0)
freq = 0.014f0
wavelet = ricker_wavelet(srcGeometry.t[1],srcGeometry.dt[1],freq);  # 15 Hz wavelet
wavelet = low_filter(wavelet, srcGeometry.dt[1]; fmin=4f0, fmax=freq*1000f0);

q = judiVector(srcGeometry, wavelet)

# Setup operators
Pr = judiProjection(recGeometry)
A_inv = judiModeling(model;)
Ps = judiProjection(srcGeometry)

F_wave = Pr*A_inv*adjoint(Ps)

####################################### Make observation ###################
snr = 30

#F = x -> reshape(joEye(prod(n);DDT=Float32,RDT=Float32)*vec(x),n)
#dobs = F(m) 
F_wave(m,q[2])
F = x -> F_wave(x,q)
snr_scale = 10^(-snr/20)
d_sim = F(m) 
Random.seed!(123);
e = randn(size(d_sim.data[1]));
e = judiVector(d_sim.geometry, e);
e = e*snr_scale*norm(d_sim)/norm(e)
dobs = d_sim + e;


####################################### Optimization ######################
proj(x) = reshape(median([vec(mmin) vec(x) vec(mmax)]; dims=2),n)
ls = BackTracking(order=3, iterations=10)

# Experiment configurations
opt_param = "v"
device = cpu

# setup generator network g(z)->image by reversing the normalizing flow t(image)->z
z_init = m0;
if opt_param == "z"
    global device = gpu 

    # Load pretrainedn normalizing flow T
    net_path = "data/clip_norm=10.0_depth=5_e=59_lr=0.0001_nc_hidden=512_nscales=6_ntrain=2640_nx=512_ny=256_α=0.1_αmin=0.01.jld2"
    Params = JLD2.jldopen(net_path, "r")["Params"];
    Rrn_here = JLD2.jldopen(net_path, "r")["Rrn_here"];

    copy!(Random.default_rng(), Rrn_here);
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
    G = Glow(nc, nc_hidden, depth, nscales; logdet=false, opt=opts) #|> gpu

    set_params!(G, gpu(Params));
    G = G |> device

    # check generative samples are good so that loading went well. 
    G.forward(randn(Float32,nx,ny,1,1) |> device);
    gen = G.inverse(randn(Float32,nx,ny,1,1)|> device)

    fig=figure(figsize=(15,7));
    imshow(gen[:,:,1,1]');

    tight_layout()
    fig_name = @strdict
    safesave(joinpath(plot_path,savename(fig_name; digits=6)*"_gen.png"), fig); close(fig)


    global G = reverse(G);#device;
    #global z_init = 0f0 .* R(G.inverse(R(get_v(z_init))))
    global z_init = G.inverse(get_v(z_init))

    opt_var_init = L"Initial guess $G_{\theta}(z_{0})$"
    opt_var_final = L"MAP $G_{\theta}(z^\ast)$"
    opt_var_init_data = L"Initial guess $F(G_{\theta}(z_{0}))$"
    opt_var_final_data = L"MAP $F(G_{\theta}(z^\ast))$"

    function S(v)
         m = (1f0 ./ (vec(v))).^2f0
    end

    function get_z(m)
        z = collect(reshape(m, n[1],n[2], 1,1))
        G.inverse(get_v(z |> device))
    end
else
    G = x -> x#reshape(joEye(prod(n);DDT=Float32,RDT=Float32)*vec(x),size(x))
    opt_var_init = L"Initial guess $v_{0}$"
    opt_var_final = L"MAP $m^{\ast}$"
    opt_var_init_data = L"Initial guess $F(v_{0})$"
    opt_var_final_data = L"MAP $F(v^{\ast})$"
    function S(v)
        m = 1f0 .* vec(v)
    end
    function get_z(m)
        m
    end
end

function get_m(z)
    x = G(z |> device)[:,:,1,1] |>cpu
    m = S(x)
end

fig=figure(figsize=(15,7));
title("MAP / ADAM")
subplot(1,2,1); title(opt_var_init)
imshow(reshape(get_v(get_m(z_init)),n)'|>cpu;cmap="cet_rainbow4",vmin=NewMin,vmax=NewMax,interpolation="none"); colorbar()

subplot(1,2,2); title(opt_var_init)
imshow(v0[:,:,1,1]'|>cpu;cmap="cet_rainbow4",interpolation="none",vmin=NewMin,vmax=NewMax); colorbar()

tight_layout()
fig_name = @strdict
safesave(joinpath(plot_path,savename(fig_name; digits=6)*"_map.png"), fig); close(fig)

# ls problem parameterized by latentz
function f(z)
    m = get_m(z) 
    dpred = F(m)
    global misfit = .5f0/(nsrc^2f0) * norm(dpred-dobs)^2f0
    global prior = λ*norm(z)^2f0/length(z)
    global fval = misfit + prior
    @show misfit, prior, fval
    return fval
end

# Starting point z and predicted data
z = copy(z_init);
dpred_0 = F_wave(get_m(z),q[1])

λ = 0
n_epochs = 3
plot_every = 1
fval = 0

losses = []
psnrs = []
l2_loss = []
for i in 1:n_epochs
    #println("$(i)/$(n_epochs)")
    @time grad = gradient(()->f(z), Flux.params(z))
    gradient_t = grad.grads[z]
    p = -gradient_t/norm(gradient_t, Inf)

    # linesearch THIS ENFORCES THAT THE OPTIMIZATION MUST BE ON M
    function ϕ(α)
        m = proj(get_m(Float32.(z .+ α * p)))
        dpred = F(m)
        global misfit = .5f0/(nsrc^2f0) * norm(dpred-dobs)^2f0
        @show α, misfit
        return misfit
    end
    step, fval = ls(ϕ, 1f-1, fval, dot(gradient_t, p))

    new_m = proj(get_m(z .+ Float32(step) .* p))
    z = get_z(new_m) 
    
    v_curr = reshape(get_v(get_m(z)),n)
    psnr = round(assess_psnr(v_curr,v);digits=3)
    l2 = round(norm(v_curr-v)^2 ;digits=3)
    append!(psnrs, psnr)
    append!(l2_loss, l2)
    append!(losses, fval)

    if mod(i,plot_every) == 0
        fig=figure(figsize=(7,10));
        subplot(3,1,1); plot(losses); ylabel("Objective f"); xlabel("Parameter update")
        ;title("final f=$(losses[end])")
        subplot(3,1,2); plot(psnrs); ylabel("PSNR metric"); xlabel("Parameter update")
         ;title("final psnr=$(psnrs[end])")
         subplot(3,1,3); plot(l2_loss); ylabel("L2 metric"); xlabel("Parameter update")
         ;title("final l2_loss=$(l2_loss[end])")
        tight_layout()
        fig_name = @strdict i snr nsrc
        safesave(joinpath(plot_path,savename(fig_name; digits=6)*"_log.png"), fig); close(fig)

        fig=figure(figsize=(21,8));
        imshow(p[:,:,1,1]'|>cpu; cmap="cet_rainbow4",extent=model_extent,interpolation="none"); colorbar()
        tight_layout()
        fig_name = @strdict i snr nsrc 
        safesave(joinpath(plot_path,savename(fig_name; digits=6)*"_grad.png"), fig); close(fig)


        # Important look at the overfitting results. Why are they not perfect?
        fig=figure(figsize=(21,8));
        title("MAP epoch = $(i)")
        subplot(1,3,1); title(opt_var_init)
        imshow(v0'|>cpu; cmap="cet_rainbow4",vmin=NewMin,vmax=NewMax,extent=model_extent,interpolation="none"); #colorbar()
        xlabel("X [m]"); ylabel("Depth [m]");

        subplot(1,3,2); title(opt_var_final*" PSNR=$(psnrs[end])")
        imshow(v_curr'|>cpu; cmap="cet_rainbow4",vmin=NewMin,vmax=NewMax,extent=model_extent,interpolation="none"); #colorbar()
        xlabel("X [m]"); ylabel("Depth [m]");

        subplot(1,3,3); title(L"Ground truth $v_{gt}$")
        imshow(v';cmap="cet_rainbow4",vmin=NewMin,vmax=NewMax,extent=model_extent,interpolation="none"); #colorbar()
        xlabel("X [m]"); ylabel("Depth [m]");

        tight_layout()
        fig_name = @strdict i snr nsrc 
        safesave(joinpath(plot_path,savename(fig_name; digits=6)*"_map.png"), fig); close(fig)

        dpred   = F_wave(get_m(z),q[1])

        fig = figure(figsize=(12,8));
        title("MAP  epoch = $(n_epochs)")
        subplot(1,3,3); title(L"Observed data $F(v_{gt}) + \eta$")

        data_plot = dobs.data[1]
        a = quantile(abs.(vec(data_plot)), 90/100)
        imshow(data_plot; vmin=-a, vmax=a,interpolation="none", cmap="PuOr", aspect="auto"); #colorbar()
        xlabel("Receiver index"); ylabel("Time [milliseconds]");

        data_plot = dpred_0.data[1]
        subplot(1,3,1); title(opt_var_init_data)
        imshow(data_plot; vmin=-a, vmax=a,interpolation="none", cmap="PuOr", aspect="auto"); #colorbar()
        #imshow(dpred_0'; vmin=-1, vmax=1, cmap="PuOr", aspect="auto"); colorbar()
        xlabel("Receiver index"); ylabel("Time [milliseconds]");

        data_plot = dpred.data[1]
        subplot(1,3,2); title(opt_var_final_data)
        imshow(data_plot; vmin=-a, vmax=a,interpolation="none", cmap="PuOr", aspect="auto"); #colorbar()
        #imshow(dpred'; vmin=-1, vmax=1, cmap="PuOr", aspect="auto"); colorbar()
        xlabel("Receiver index"); ylabel("Time [milliseconds]");

        tight_layout()
        fig_name = @strdict i  snr  nsrc 
        safesave(joinpath(plot_path,savename(fig_name; digits=6)*"_map_data.png"), fig); close(fig)
    end
end

