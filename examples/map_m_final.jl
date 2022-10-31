#export CUDA_VISIBLE_DEVICES=5
#CUDA_VISIBLE_DEVICES=7 nohup julia --project=.  examples/map_m_final.jl &
#using DrWatson
#@quickactivate :NormalizingFlow3D
#import Pkg; Pkg.instantiate()

#CUDA_VISIBLE_DEVICES=1 julia --project=.

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
using ChainRules

font_size = 12
PyPlot.rc("font", family="serif", size=font_size); PyPlot.rc("xtick", labelsize=font_size); PyPlot.rc("ytick", labelsize=font_size);
PyPlot.rc("axes", labelsize=font_size)    # fontsize of the x and y labels

# Plotting path
experiment_name = "map_2d"
plot_path = "plots/map_2d"
data_path = "../NormalizingFlow3D.jl/data/compass_volume.jld2"


function get_v(m)
    sqrt.(1f0./m)
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
d = (10., 10.) # true d
o = (0., 0.)

# Velocity [km/s]
v  = X_gt


#NewMin = 1.480f0
water_vel = 1.480f0
#NewMax = 4.5f0
idx_wb = 21

function water_mute(v)
    return hcat(water_vel * ones(Float32, n[1], idx_wb), 0f0 * zeros(Float32, n[1], n[2]-idx_wb)) + hcat(0f0 * zeros(Float32, n[1], idx_wb), ones(Float32, n[1], n[2]-idx_wb)) .* v
end

fig=figure(figsize=(15,7));
imshow(v[:,:]'); colorbar()
tight_layout()
fig_name = @strdict
safesave(joinpath(plot_path,savename(fig_name; digits=6)*"_v_get.png"), fig); close(fig)


NewMin = water_vel#minimum(v)#1.480f0
NewMax = maximum(v)#1.480f0
range_v = LinRange(minimum(v),NewMax,ny-idx_wb+1)

# v0 = water_vel .* ones(Float32,n)
# for i in idx_wb:ny
#     v0[:,i] .= range_v[i-idx_wb+1]
# end
v0 = imfilter(v, Kernel.gaussian(15f0))

v = water_mute(v)
v0 = water_mute(v0)

# fig=figure(figsize=(15,7));
# imshow(v[:,:]'); colorbar()
# imshow(v0[:,:]'); colorbar()
# tight_layout()
# fig_name = @strdict
# safesave(joinpath(plot_path,savename(fig_name; digits=6)*"_v_get.png"), fig); close(fig)

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
nsrc = 64    # number of sources
model = Model(n, d, o, m;)

#' ## Create source and receivers positions at the surface
# Set up receiver geometry
nxrec = 512
xrec = range(d[1], stop=(n[1]-1)*d[1], length=nxrec)
yrec = 0f0 # WE have to set the y coordiante to zero (or any number) for 2D modeling
zrec = range(d[1], stop=d[1], length=nxrec)

# receiver sampling and recording time
#timeD = 7000f0   # receiver recording time [ms] # try going down to 600
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

srcGeometry = Geometry(xsrc, ysrc, zsrc; dt=dtD, t=timeD)

# setup wavelet 
freq = 0.015f0
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

F = x -> F_wave(x,q)
snr_scale = 10^(-snr/20)
d_sim = F(m) 
Random.seed!(123);
e = randn(size(d_sim.data[1]));
e = judiVector(d_sim.geometry, e);
e = e*snr_scale*norm(d_sim)/norm(e)
d_obs = d_sim + e;

####################################### Optimization ######################
device = cpu
proj(x) = reshape(median([vec(mmin) vec(x) vec(mmax)]; dims=2),n)
ls = BackTracking(order=3, iterations=10)

# Experiment configurations
opt_param = "z"

# setup generator network g(z)->image by reversing the normalizing flow t(image)->z
shift = 1.0f0
z_init = m0;
net_epoch = -1
if opt_param == "z"
    global device = gpu 

    function S(v)
         m = (1f0 ./ (vec(v))).^2f0
    end

    function get_z(m)
        z = collect(reshape(m, n[1],n[2], 1,1))
        G.inverse(get_v(z |> device))
    end

    # Load pretrainedn normalizing flow T
    #net_path = "data/clip_norm=10.0_depth=5_e=59_lr=0.0001_nc_hidden=512_nscales=6_ntrain=2640_nx=512_ny=256_α=0.1_αmin=0.01.jld2"
    global net_epoch = 22
    net_path = "data/clip_norm=10.0_depth=5_e=22_lr=0.0002_nc_hidden=512_nscales=6_ntrain=10768_nx=512_ny=256_α=0.1_αmin=0.01.jld2"
    
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
    imshow(gen[:,:,1,1]'|>cpu);

    tight_layout()
    fig_name = @strdict
    safesave(joinpath(plot_path,savename(fig_name; digits=6)*"_gen.png"), fig); close(fig)

    global G = reverse(G);
    #global z_init = 0f0 .* R(G.inverse(R(get_v(z_init))))
    global z_init = shift .* get_z(z_init)# G.inverse(get_v(z_init|> device))

    opt_var_init = L"Initial guess $G_{\theta}(z_{0})$"
    opt_var_final = L"MAP $G_{\theta}(z^\ast)$"
    opt_var_init_data = L"Initial guess $F(G_{\theta}(z_{0}))$"
    opt_var_final_data = L"MAP $F(G_{\theta}(z^\ast))$"
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
    x = G(z |> device) |>cpu
    m = S(x)
end

# fig=figure(figsize=(15,7));
# title("l2 ball $(shift)")
# subplot(1,2,1); title(opt_var_init)
# imshow(reshape(get_v(get_m(z_init)),n)'|>cpu;cmap="cet_rainbow4",vmin=NewMin,vmax=NewMax,interpolation="none"); colorbar()

# subplot(1,2,2); title(opt_var_init)
# imshow(v0[:,:,1,1]'|>cpu;cmap="cet_rainbow4",interpolation="none",vmin=NewMin,vmax=NewMax); colorbar()

# tight_layout()
# fig_name = @strdict shift
# safesave(joinpath(plot_path,savename(fig_name; digits=6)*"_map.png"), fig); close(fig)

# ls problem parameterized by latentz
# function f(z,i_src, f_wave_src)
#     m = get_m(z) 
#     dpred = f_wave_src(m)#F_wave[i_src](m,q[i_src])
#     global misfit = .5f0/(nsrc^2f0) * norm(dpred-dobs[i_src])^2f0
#     global prior = λ*norm(z)^2f0/length(z)
#     global fval = misfit + prior
#     @show misfit, prior, fval
#     return fval
# end

function f(z,i_src)
    m = get_m(z) 
    #dpred = F_wave[i_src](m,q[i_src])
    #dpred = F_sub(m,q)
    Floc, qloc, dloc = ChainRules.@ignore_derivatives F_wave[i_src], q[i_src], d_obs[i_src]
    dpred = Floc(m,qloc)
    global fval = 0.5f0 * norm(dpred-dloc)^2f0
    fval
end

# Starting point z and predicted data
z = copy(z_init);
dpred_0 = F_wave(get_m(z),q[[1]])
#dpred_0 = F(get_m(z))
#t = q[[19]].data
#q[[1]].data

λ = 0
batchsize = 8
n_epochs = 5
plot_every = 1
fval = 0

#map_lr = 6f-3 #seems to work find in fwi
map_lr = -6f-3 #seems to work find in fwi
opt = ADAM(map_lr)

losses = []
psnrs = []
l2_loss = []
for i in 1:n_epochs

    println("$(i)/$(n_epochs)")
    i_src = randperm(d_obs.nsrc)[1:batchsize]

    curr_v = G.forward(z)
    curr_m = (1f0 ./ (curr_v)).^2f0
    fval, gradient_fwi = fwi_objective(Model(n, d, o, curr_m[:,:,1,1]|>cpu), q[i_src], d_obs[i_src])

    gradient_dm = gradient_fwi.data |> device
    dv = gradient_dm ./ (-2 .* curr_v .^(-3))
    gradient_t = G.backward(dv, curr_v)[1]

    #grad = gradient(()->f(z,i_src), Flux.params(z))
    #gradient_t = grad.grads[z]

    # fig=figure(figsize=(21,8));
    # subplot(1,2,1); title("manual grad")
    # imshow(dz[:,:,1,1]'|>cpu; cmap="cet_rainbow4",extent=model_extent,interpolation="none"); colorbar()
    # xlabel("X [m]"); ylabel("Depth [m]");

    # subplot(1,2,2); title("AD grad")
    # imshow(dz_AD[:,:,1,1]'|>cpu; cmap="cet_rainbow4",extent=model_extent,interpolation="none"); colorbar()
    # xlabel("X [m]"); ylabel("Depth [m]");

    # tight_layout()
    # fig_name = @strdict i snr nsrc shift net_epoch map_lr
    # safesave(joinpath(plot_path,savename(fig_name; digits=6)*"_map.png"), fig); close(fig)

    #Flux.update!(opt, z, grad.grads[z])

    p = -gradient_t/norm(gradient_t, Inf)

    # fig = figure(figsize=(10,4))
    # plot_simage(p',d;new_fig=false)
    # fig_name = @strdict snr nsrc shift net_epoch map_lr
    # safesave(joinpath(plot_path,savename(fig_name; digits=6)*"_grad.png"), fig); close(fig)

    # linesearch THIS ENFORCES THAT THE OPTIMIZATION MUST BE ON M
    function ϕ(α)
        m = proj(get_m(Float32.(z .+ α * p)))
        dpred = F_wave[i_src](m,q[i_src])
        global misfit = .5f0 * norm(dpred-d_obs[i_src])^2f0
        @show α, misfit
        return misfit
    end
    step, fval = ls(ϕ, 1f-1, fval, dot(gradient_t, p))

    new_m = proj(get_m(z .+ Float32(step) .* p))
    global z = get_z(new_m) 
    
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
        fig_name = @strdict i snr nsrc shift net_epoch map_lr
        safesave(joinpath(plot_path,savename(fig_name; digits=6)*"_log.png"), fig); close(fig)

        # fig=figure(figsize=(21,8));
        # imshow(p[:,:,1,1]'|>cpu; cmap="cet_rainbow4",extent=model_extent,interpolation="none"); colorbar()
        # tight_layout()
        # fig_name = @strdict i snr nsrc  shift
        # safesave(joinpath(plot_path,savename(fig_name; digits=6)*"_grad.png"), fig); close(fig)

        # Important look at the overfitting results. Why are they not perfect?
        fig=figure(figsize=(21,8));
        title("MAP epoch = $(i)")
        subplot(1,3,1); title(opt_var_init)
        imshow(reshape(get_v(get_m(z_init)),n)'|>cpu; cmap="cet_rainbow4",vmin=NewMin,vmax=NewMax,extent=model_extent,interpolation="none"); #colorbar()
        xlabel("X [m]"); ylabel("Depth [m]");

        subplot(1,3,2); title(opt_var_final*" PSNR=$(psnrs[end])")
        imshow(v_curr'|>cpu; cmap="cet_rainbow4",vmin=NewMin,vmax=NewMax,extent=model_extent,interpolation="none"); #colorbar()
        xlabel("X [m]"); ylabel("Depth [m]");

        subplot(1,3,3); title(L"Ground truth $v_{gt}$")
        imshow(v';cmap="cet_rainbow4",vmin=NewMin,vmax=NewMax,extent=model_extent,interpolation="none"); #colorbar()
        xlabel("X [m]"); ylabel("Depth [m]");

        tight_layout()
        fig_name = @strdict i snr nsrc shift net_epoch map_lr freq
        safesave(joinpath(plot_path,savename(fig_name; digits=6)*"_map.png"), fig); close(fig)

        dpred   = F_wave(get_m(z),q[1])
        #dpred   = F(get_m(z))
        fig = figure(figsize=(12,8));
        title("MAP  epoch = $(n_epochs)")
        subplot(1,3,3); title(L"Observed data $F(v_{gt}) + \eta$")

        data_plot = d_obs.data[1]
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
        fig_name = @strdict i  snr  nsrc shift map_lr freq
        safesave(joinpath(plot_path,savename(fig_name;  digits=6)*"_map_data.png"), fig); close(fig)
    end
    # save_dict = @strdict i opt_param shift z_init z n_epochs map_lr λ losses l2_loss psnrs  
    # safesave(
    #  datadir("map-2d", savename(save_dict, "jld2"; digits=6)),
    #  save_dict;
    # )
end

save_dict = @strdict opt_param shift z_init z n_epochs map_lr λ losses l2_loss psnrs  
safesave(
 datadir("map-2d", savename(save_dict, "jld2"; digits=6)),
 save_dict;
)

