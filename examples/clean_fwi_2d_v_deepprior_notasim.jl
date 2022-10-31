#CUDA_VISIBLE_DEVICES=1 nohup julia --project=. examples/clean_fwi_2d_v_deepprior_notasim.jl
#CUDA_VISIBLE_DEVICES=1 nohup julia --project=. examples/clean_fwi_2d_v_deepprior_notasim.jl
#CUDA_VISIBLE_DEVICES=4 julia --project=.

using InvertibleNetworks, Flux, GlowInvertibleNetwork



using LinearAlgebra,  PyPlot,  Random
using DrWatson
using JUDI 
using JLD2
using Images
using PyPlot 
using Statistics 
using JOLI 
using ImageQualityIndexes
using SlimPlotting
using SlimOptim
using Flux 

# Plotting path
experiment_name = "map_2d"
plot_path = "plots/map_2d"

@load  "map_fwi_test_v.jld2" v 

# Set up model structure
n = size(v) # (x,y,z) or (x,z)
d = (10., 10.) # true d
o = (0., 0.)

water_vel = 1.480f0
NewMin = water_vel
NewMax = maximum(v)
idx_wb = 21

function water_mute(v)
    return hcat(water_vel * ones(Float32, n[1], idx_wb), 0f0 * zeros(Float32, n[1], n[2]-idx_wb)) + hcat(0f0 * zeros(Float32, n[1], idx_wb), ones(Float32, n[1], n[2]-idx_wb)) .* v
end

function get_v(m)
    sqrt.(1f0./m)
end

function get_m(v)
    m = (1f0 ./ (v)).^2f0
end

replace_nan(v) = map(x -> isnan(x) ? zero(x) : x, v)


# v0 = water_vel .* ones(Float32,n)
# range_v = LinRange(minimum(v),NewMax,ny-idx_wb+1)
# for i in idx_wb:ny
#     v0[:,i] .= range_v[i-idx_wb+1]
# end
v0 = imfilter(v, Kernel.gaussian(15f0))

v = water_mute(v)
v0 = water_mute(v0)

# Bound constraints
vmin = ones(Float32, n) .* 1.3f0#water_vel
vmax = ones(Float32, n) .* 6.5f0#NewMax

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
model = Model(n, d, o, m;)
model0 = Model(n, d, o, m0;)


#' ## Create source and receivers positions at the surface
# Set up receiver geometry
nxrec = 512
xrec = range(d[1], stop=(n[1]-1)*d[1], length=nxrec)
yrec = 0f0 # WE have to set the y coordiante to zero (or any number) for 2D modeling
zrec = range(d[1], stop=d[1], length=nxrec)

# receiver sampling and recording time
timeD = 3000f0   # receiver recording time [ms] # try going down to 600
dtD = 4f0    # receiver sampling interval [ms]

data_extent = (0, nxrec, timeD, 0)
model_extent = (0,(n[1]-1)*d[1],(n[2]-1)*d[2],0)

# Set up receiver structure
recGeometry = Geometry(xrec, yrec, zrec; dt=dtD, t=timeD, nsrc=nsrc)

# Set up source structure
xsrc = convertToCell(range(d[1],stop=(n[1]-1)*d[1], length=nsrc))
ysrc = convertToCell(range(0f0,stop=0f0,length=nsrc))
zsrc = convertToCell(range((idx_wb-1)*d[2]-2f0,stop=(idx_wb-1)*d[2]-2f0,length=nsrc))
src_geometry = Geometry(xsrc, ysrc, zsrc; dt=dtD, t=timeD)

# setup wavelet 
freq = 0.015f0
wavelet = ricker_wavelet(src_geometry.t[1],src_geometry.dt[1],freq);  # 15 Hz wavelet
wavelet = low_filter(wavelet, src_geometry.dt[1]; fmin=4f0, fmax=freq*1000f0);

q = judiVector(src_geometry, wavelet)

# Setup operators
Pr = judiProjection(recGeometry)
A_inv = judiModeling(model;)
Ps = judiProjection(src_geometry)

F_wave = Pr*A_inv*adjoint(Ps)

####################################### Make observation ###################
snr = 30

F = x -> F_wave(x,q)
snr_scale = 10^(-snr/20)
d_sim = F(m) 
Random.seed!(123);
e = randn(size(d_sim.data[1]));
e = judiVector(d_sim.geometry, e);
e = e*snr_scale*norm(d_sim)/norm(e)
d_obs = d_sim + e;

####################################### get deep prior observation ###################
device = gpu
@load "data/clip_norm=10.0_depth=5_e=120_lr=0.0002_nc_hidden=512_nscales=6_ntrain=10768_nx=512_ny=256_α=0.1_αmin=0.01.jld2"
nx_g = nx
ny_g = ny
N_g = nx_g*nx_g
n_patches = 1

copy!(Random.default_rng(), Rrn_here);
# Create multiscale network
opts = GlowOptions(; cl_activation=SigmoidNewLayer(0.5f0),
                    cl_affine=true,
                    init_cl_id=true,
                    conv1x1_nvp=false,
                    init_conv1x1_permutation=true,
                    conv1x1_orth_fixed=true,
                    T=Float32)
G = Glow(1, nc_hidden, depth, nscales; opt=opts)
set_params!(G, gpu(Params));
G = G |> device

# check generative samples are good so that loading went well. 
G.forward(randn(Float32,nx_g,ny_g,1,1) |> device);
gen = G.inverse(randn(Float32,nx_g,ny_g,1,1)|> device)

fig=figure(figsize=(15,7));
imshow(gen[:,:,1,1]'|>cpu);
tight_layout()
fig_name = @strdict e
safesave(joinpath(plot_path,savename(fig_name; digits=6)*"_gen.png"), fig); close(fig)

λ = 0.1f0
####################################### Optimization ######################
v_i = copy(v0)
F0 = judiModeling(deepcopy(model0), src_geometry, d_obs.geometry)

proj(x) = reshape(median([vec(vmin) vec(x) vec(vmax)]; dims=2),n)
#proj(x) = reshape(median([vec(vmin) vec(x) vec(vmax)]; dims=2),n)
ls = BackTracking(order=3, iterations=10)

# Starting point z and predicted data
dpred_0 = F_wave(model0,q[1])

batchsize = 16
n_epochs = 50
plot_every = 1

map_lr = 4f-3 #seems to work find in fwi


losses = []
psnrs = []
l2_loss = []
losses_prior = []
Random.seed!(123); #need random source sampling to be the same
for i in 1:n_epochs
    println("$(i)/$(n_epochs)")
    i_src = randperm(d_obs.nsrc)[1:batchsize]
    model_i = Model(n, d, o,  get_m(v_i);)
    fval, gradient_dm = fwi_objective(model_i, q[i_src], d_obs[i_src])
    gradient_fwi = gradient_dm.data ./ (-2 .* v_i .^(-3))

    gradient_total = gradient_fwi #+ λ*grad_prior
    #gradient =  λ*grad_prior

    p = -gradient_total/norm(gradient_total, Inf)

    #global v_i = proj(v_i .+ 0.01 .* p)

    # fig=figure(figsize=(21,8));
    # subplot(1,3,1); title("starting v0 p_{\theta}(x)=$(prior)")
    # imshow(v0'; cmap="cet_rainbow4",vmin=NewMin,vmax=NewMax,extent=model_extent,interpolation="none");# colorbar()
    # xlabel("X [m]"); ylabel("Depth [m]");

    # subplot(1,3,3); title("next vi p_{\theta}(x)=$(prior)")
    # imshow(v_i';cmap="cet_rainbow4",vmin=NewMin,vmax=NewMax,extent=model_extent,interpolation="none"); #colorbar()
    # xlabel("X [m]"); ylabel("Depth [m]");

    # tight_layout()
    # fig_name = @strdict i snr nsrc freq batchsize λ
    # safesave(joinpath(plot_path,savename(fig_name; digits=6)*"_map.png"), fig); close(fig)

    # linesearch 
    function ϕ(α)
        F0.model.m .= get_m(proj(v_i .+ α * p))
        misfit = .5*norm(F0[i_src]*q[i_src] - d_obs[i_src])^2
        @show α, misfit
        return misfit
    end
    step_t, fval = ls(ϕ, 1f0, fval, dot(gradient_total, p))

    global v_i = proj(v_i .+ step_t .* p)

    #Prior on velocity
    opt = Flux.Optimiser([ADAM(map_lr)])
    #for j in 1:50
        z_probe = v_i
        grad_prior     = zeros(Float32,size(z_probe)) |> device
        grad_prior_norm = ones(Float32,size(z_probe)) |> device
        nx_z, ny_z = size(grad_prior)[1:2]
        
        x_coord = [rand(1:nx_z-nx_g+1) for l =1:n_patches]
        y_coord = [rand(1:ny_z-ny_g+1) for l =1:n_patches]
        inds = [(x_coord[l]:x_coord[l]+nx_g-1,y_coord[l]:y_coord[l]+ny_g-1,1:1,1:1) for l = 1:n_patches]
        
        z_patches = zeros(Float32,nx_g,ny_g,1,n_patches) |> device
        for k in 1:n_patches
            z_patches[:,:,:,k] = z_probe[inds[k]...]
        end

        zz, lgdet = G(z_patches);

        z_patches_grads = G.backward(zz/n_patches,zz)[1] 
        z_patches_grads = replace_nan(z_patches_grads)
        for k in 1:n_patches
            grad_prior_norm[inds[k]...] += ones(size(z_patches_grads[:,:,:,k:k]))|>gpu
            grad_prior[inds[k]...] += z_patches_grads[:,:,:,k:k]
        end

        prior = norm(zz)^2/(n_patches*N_g) - lgdet / N_g
    #    grad_prior = grad_prior ./ grad_prior_norm |> cpu

    #    Flux.update!(opt,v_i,grad_prior)
    #    println(prior)
    #    append!(losses_prior,prior)
    #end
    
    v_curr = v_i
    psnr = round(assess_psnr(v_curr,v);digits=3)
    l2 = round(norm(v_curr-v)^2 ;digits=3)

    @show fval, prior

    append!(psnrs, psnr);append!(losses_prior, prior)
    append!(l2_loss, l2)
    append!(losses, fval)
    if mod(i,plot_every) == 0
        fig=figure(figsize=(7,10));
        subplot(4,1,1); plot(losses); ylabel("Objective f"); xlabel("Parameter update")
        ;title("final f=$(losses[end])")
        subplot(4,1,2); plot(losses_prior); ylabel("prior"); xlabel("Parameter update")
    
        subplot(4,1,3); plot(psnrs); ylabel("PSNR metric"); xlabel("Parameter update")
        ;title("final psnr=$(psnrs[end])")
        subplot(4,1,4); plot(l2_loss); ylabel("L2 metric"); xlabel("Parameter update")
        ;title("final l2_loss=$(l2_loss[end])")
        tight_layout()
        fig_name = @strdict i snr nsrc freq batchsize λ 
        safesave(joinpath(plot_path,savename(fig_name; digits=6)*"_log.png"), fig); close(fig)

        # Important look at the overfitting results. Why are they not perfect?
        fig=figure(figsize=(21,8));
        title("MAP epoch = $(i)")
        subplot(1,3,1); title("starting v0")
        imshow(v0'; cmap="cet_rainbow4",vmin=NewMin,vmax=NewMax,extent=model_extent,interpolation="none");# colorbar()
        xlabel("X [m]"); ylabel("Depth [m]");

        subplot(1,3,2); title("curr v_i PSNR=$(psnrs[end])")
        imshow(v_curr'; cmap="cet_rainbow4",vmin=NewMin,vmax=NewMax,extent=model_extent,interpolation="none"); #colorbar()
        xlabel("X [m]"); ylabel("Depth [m]");

        subplot(1,3,3); title(L"Ground truth $v_{gt}$")
        imshow(v';cmap="cet_rainbow4",vmin=NewMin,vmax=NewMax,extent=model_extent,interpolation="none"); #colorbar()
        xlabel("X [m]"); ylabel("Depth [m]");

        tight_layout()
        fig_name = @strdict i snr nsrc freq batchsize λ
        safesave(joinpath(plot_path,savename(fig_name; digits=6)*"_map.png"), fig); close(fig)

        dpred   = F_wave(get_m(v_i),q[1])
        fig = figure(figsize=(12,8));
        title("MAP  epoch = $(n_epochs)")
        subplot(1,3,3); title(L"Observed data $F(v_{gt}) + \eta$")
        data_plot = d_obs.data[1]
        a = quantile(abs.(vec(data_plot)), 90/100)
        imshow(data_plot; vmin=-a, vmax=a,interpolation="none", cmap="PuOr", aspect="auto"); #colorbar()
        xlabel("Receiver index"); ylabel("Time [milliseconds]");

        data_plot = dpred_0.data[1]
        subplot(1,3,1); title(L"initial data $F(v_{0}) + \eta$")
        imshow(data_plot; vmin=-a, vmax=a,interpolation="none", cmap="PuOr", aspect="auto"); #colorbar()
        xlabel("Receiver index"); ylabel("Time [milliseconds]");

        data_plot = dpred.data[1]
        subplot(1,3,2); title(L"current data $F(v_{i}) + \eta$")
        imshow(data_plot; vmin=-a, vmax=a,interpolation="none", cmap="PuOr", aspect="auto"); #colorbar()
        xlabel("Receiver index"); ylabel("Time [milliseconds]");

        tight_layout()
        fig_name = @strdict i snr nsrc freq batchsize λ
        safesave(joinpath(plot_path,savename(fig_name;  digits=6)*"_map_data.png"), fig); close(fig)
    end

    save_dict = @strdict i v_i n_epochs λ losses l2_loss psnrs  losses_prior
    safesave(
     datadir("map-2d", savename(save_dict, "jld2"; digits=6)),
     save_dict;
    )
end



# save_dict = @strdict v_i n_epochs λ losses l2_loss psnrs  losses_prior
# safesave(
#  datadir("map-2d", savename(save_dict, "jld2"; digits=6)),
#  save_dict;
# )

