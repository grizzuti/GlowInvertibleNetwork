export GlowLowDim, GlowLowDimOptions

struct GlowLowDim{T} <: InvertibleNetwork
    depth::Int64
    FS::AbstractArray{FlowStep{T},1}
    logdet::Bool
end

@Flux.functor GlowLowDim

struct GlowLowDimOptions{T<:Real}
    fs_options::FlowStepOptions{T}
end

function GlowLowDimOptions(;
                       actnorm1::Bool=true,
                       actnorm2::Bool=true,
                       weight_std1::Real=0.05,
                       weight_std2::Real=0.05,
                       weight_std3::Union{Nothing,Real}=nothing,
                       logscale_factor::Real=3.0,
                       cl_activation::Union{Nothing,InvertibleNetworks.ActivationFunction}=SigmoidNewLayer(),
                       cl_affine::Bool=true,
                       init_cl_id::Bool=true,
                       conv1x1_nvp::Bool=true,
                       init_conv1x1_permutation::Bool=true,
                       T::DataType=Float32)

    opt_cb = ConvolutionalBlockOptions(; k1=1, p1=0, s1=1, actnorm1=actnorm1, k2=1, p2=0, s2=1, actnorm2=actnorm2, k3=1, p3=0, s3=1, weight_std1=weight_std1, weight_std2=weight_std2, weight_std3=weight_std3, logscale_factor=logscale_factor, init_zero=init_cl_id, T=T)
    opt_cl = CouplingLayerAffineOptions(; options_convblock=opt_cb, activation=cl_activation, affine=cl_affine)
    opt_conv1x1 = Conv1x1genOptions(; nvp=conv1x1_nvp, init_permutation=init_conv1x1_permutation, T=T)

    return GlowLowDimOptions{T}(FlowStepOptions{T}(opt_cl, opt_conv1x1))

end

function GlowLowDim(nc::Int64, nc_hidden::Int64, depth::Int64; logdet::Bool=true, opt::GlowLowDimOptions{T}=GlowLowDimOptions()) where T

    FS = Array{FlowStep{T},1}(undef, depth)
    for k = 1:depth
        FS[k] = FlowStep(nc, nc_hidden; logdet=logdet, opt=opt.fs_options)
    end
    return GlowLowDim{T}(depth,FS,logdet)

end

function forward(X::AbstractArray{T,4}, G::GlowLowDim{T}) where T

    # Initialize logdet
    G.logdet && (logdet = T(0))

    # Loop over depth
    for k = 1:G.depth
        G.logdet ? ((X, lgdt) = G.FS[k].forward(X)) : (X = G.FS[k].forward(X))
        G.logdet && (logdet += lgdt)
    end
    Y = X

    G.logdet ? (return Y, logdet) : (return Y)

end

function inverse(Y::AbstractArray{T,4}, G::GlowLowDim{T}) where T

    X = Y
    for k = G.depth:-1:1
        X = G.FS[k].inverse(X)
    end

    return X

end

function backward(ΔY::AbstractArray{T,4}, Y::AbstractArray{T,4}, G::GlowLowDim{T}) where T

    ΔX = ΔY
    X  = Y
    for k = G.depth:-1:1
        ΔX, X = G.FS[k].backward(ΔX, X)
    end

    return ΔX, X

end

function clear_grad!(G::GlowLowDim)
    for k=1:G.depth
        clear_grad!(G.FS[k])
    end
end

function get_params(G::GlowLowDim)
    p = []
    for k=1:G.depth
        push!(p, get_params(G.FS[k]))
    end
    return cat(p...; dims=1)
end

function gpu(G::GlowLowDim{T}) where T
    FS = Array{FlowStep{T},1}(undef,G.depth)
    for k=1:G.depth
        FS[k] = gpu(G.FS[k])
    end
    return GlowLowDim{T}(G.depth, FS, G.logdet)
end

function cpu(G::GlowLowDim{T}) where T
    FS = Array{FlowStep{T},1}(undef,G.depth)
    for k=1:G.depth
        FS[k] = cpu(G.FS[k])
    end
    return GlowLowDim{T}(G.depth, FS, G.logdet)
end