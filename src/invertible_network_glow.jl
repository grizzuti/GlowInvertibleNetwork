export Glow, GlowOptions

struct Glow{T,N} <: InvertibleNetwork
    depth::Int64
    nscales::Int64
    scale_dims::Array{NTuple{N,Int64},1}
    FS::AbstractArray{FlowStep{T},2}
    logdet::Bool
end

@Flux.functor Glow

struct GlowOptions{T<:Real}
    fs_options::FlowStepOptions{T}
end

function GlowOptions(; k1::Int64=3, p1::Int64=1, s1::Int64=1, actnorm1::Bool=true,
                       k2::Int64=1, p2::Int64=0, s2::Int64=1, actnorm2::Bool=true,
                       k3::Int64=3, p3::Int64=1, s3::Int64=1,
                       weight_std1::Real=0.05,
                       weight_std2::Real=0.05,
                       weight_std3::Union{Nothing,Real}=nothing,
                       logscale_factor::Real=3.0,
                       cl_activation::Union{Nothing,InvertibleNetworks.ActivationFunction}=SigmoidNewLayer(0.5),
                       cl_affine::Bool=true,
                       init_cl_id::Bool=true,
                       conv1x1_nvp::Bool=true,
                       init_conv1x1_permutation::Bool=true,
                       conv1x1_orth_fixed::Bool=true,
                       T::DataType=Float32)

    opt_cb = ConvolutionalBlockOptions(; k1=k1, p1=p1, s1=s1, actnorm1=actnorm1, k2=k2, p2=p2, s2=s2, actnorm2=actnorm2, k3=k3, p3=p3, s3=s3, weight_std1=weight_std1, weight_std2=weight_std2, weight_std3=weight_std3, logscale_factor=logscale_factor, init_zero=init_cl_id, T=T)
    opt_cl = CouplingLayerAffineOptions(; options_convblock=opt_cb, activation=cl_activation, affine=cl_affine)
    conv1x1_orth_fixed ? (opt_conv1x1=nothing) : (opt_conv1x1 = Conv1x1genOptions(; nvp=conv1x1_nvp, init_permutation=init_conv1x1_permutation, T=T))

    return GlowOptions{T}(FlowStepOptions{T}(opt_cl, opt_conv1x1))

end

function Glow(nc::Int64, nc_hidden::Int64, depth::Int64, nscales::Int64;ndims=2, logdet::Bool=true, opt::GlowOptions{T}=GlowOptions()) where T

    FS = Array{FlowStep{T},2}(undef,depth,nscales)
    channel_factor = 2^(ndims)
    nc = channel_factor*nc

    N = ndims+2
    for l = 1:nscales
        for k = 1:depth
            FS[k,l] = FlowStep(nc, nc_hidden;ndims=ndims, logdet=logdet, opt=opt.fs_options)
        end
        nc = Int(channel_factor/2)*nc
    end
    return Glow{T,N}(depth,nscales,Array{NTuple{N,Int64}}(undef,nscales),FS,logdet)

end

function forward(X::AbstractArray{T,N}, G::Glow{T}) where {T, N}

    # Original input shape
    input_shape = size(X)

    # Keeping track of intermediate scale outputs
    Yscales = Array{Any,1}(undef,G.nscales)

    # Initialize logdet
    G.logdet && (logdet = T(0))

    for l = 1:G.nscales-1 # Loop over scales
        X = squeeze(X; pattern="checkerboard")
        for k = 1:G.depth
            G.logdet ? ((X, lgdt) = G.FS[k,l].forward(X)) : (X = G.FS[k,l].forward(X))
            G.logdet && (logdet += lgdt)
        end
        X, Yl = tensor_split(X)
        G.scale_dims[l] = size(Yl)
        Yscales[l] = vec(Yl)

    end # end loop over scales

    X = squeeze(X; pattern="checkerboard")
    for k = 1:G.depth
        G.logdet ? ((X, lgdt) = G.FS[k,end].forward(X)) : (X = G.FS[k,end].forward(X))
        G.logdet && (logdet += lgdt)
    end
    G.scale_dims[G.nscales] = size(X)
    Yscales[G.nscales] = vec(X)

    # Concatenating scales
    Y = reshape(cat_scales(Yscales), input_shape)

    G.logdet ? (return Y, logdet) : (return Y)

end

function inverse(Y::AbstractArray{T,N}, G::Glow{T}) where {T, N}

    # De-concatenating scales
    Yscales = uncat_scales(Y[:], G.scale_dims)

    X = Yscales[G.nscales]
    for k = G.depth:-1:1
        X = G.FS[k,end].inverse(X)
    end
    X = unsqueeze(X; pattern="checkerboard")

    for l = (G.nscales-1):-1:1 # Loop over scales

        X = tensor_cat(X, Yscales[l])
        for k = G.depth:-1:1
            X = G.FS[k,l].inverse(X)
        end
        X = unsqueeze(X; pattern="checkerboard")

    end # end loop over scales

    return X

end

function backward(ΔY::AbstractArray{T,N}, Y::AbstractArray{T,N}, G::Glow{T}) where {T, N}

    # De-concatenating scales
    ΔYscales = uncat_scales(ΔY[:], G.scale_dims)
    Yscales  = uncat_scales(Y[:], G.scale_dims)

    ΔX = ΔYscales[end]
    X  = Yscales[end]
    for k = G.depth:-1:1
        ΔX, X = G.FS[k,end].backward(ΔX, X)
    end
    ΔX = unsqueeze(ΔX; pattern="checkerboard")
    X  = unsqueeze(X; pattern="checkerboard")

    for l = (G.nscales-1):-1:1 # Loop over scales

        ΔX = tensor_cat(ΔX, ΔYscales[l])
        X  = tensor_cat(X,   Yscales[l])
        for k = G.depth:-1:1
            ΔX, X = G.FS[k,l].backward(ΔX, X)
        end
        ΔX = unsqueeze(ΔX; pattern="checkerboard")
        X  = unsqueeze(X; pattern="checkerboard")

    end # end loop over scales

    return ΔX, X
end
function backward_inv(ΔX::AbstractArray{T, N},  X::AbstractArray{T, N}, G::Glow{T, N}) where {T, N}
    ΔYscales = Array{Any,1}(undef,G.nscales)
    Yscales = Array{Any,1}(undef,G.nscales)

    # Original input shape
    input_shape = size(X)

    for l = 1:G.nscales-1 # Loop over scales
        ΔX = squeeze(ΔX; pattern="checkerboard")
        X = squeeze(X; pattern="checkerboard")
        for k = 1:G.depth
            ΔX, X = G.FS[k,l].backward_inv(ΔX,X)
        end
        X, Yl = tensor_split(X)
        Yscales[l] = vec(Yl)

        ΔX, ΔYl = tensor_split(ΔX)
        ΔYscales[l] = vec(ΔYl)

        G.scale_dims[l] = size(Yl)
    end # end loop over scales

    ΔX = squeeze(ΔX; pattern="checkerboard")
    X = squeeze(X; pattern="checkerboard")
    for k = 1:G.depth
        ΔX, X = G.FS[k,end].backward_inv(ΔX,X)
    end
    G.scale_dims[G.nscales] = size(X)
    Yscales[G.nscales] = vec(X)
    ΔYscales[G.nscales] = vec(ΔX)

    ΔX = reshape(cat_scales(ΔYscales), input_shape)
    X = reshape(cat_scales(Yscales), input_shape)

    return ΔX, X
end


# function backward_inv(ΔX::AbstractArray{T, N},  X::AbstractArray{T, N}, G::NetworkGlow) where {T, N}
#     G.split_scales && (X_save = array_of_array(X, G.L-1))
#     G.split_scales && (ΔX_save = array_of_array(ΔX, G.L-1))
#     orig_shape = size(X)

#     for i=1:G.L
#         G.split_scales && (ΔX = G.squeezer.forward(ΔX))
#         G.split_scales && (X  = G.squeezer.forward(X))
#         for j=1:G.K
#             ΔX_, X_ = backward_inv(ΔX, X, G.AN[i, j])
#             ΔX,  X  = backward_inv(ΔX_, X_, G.CL[i, j])
#         end

#         if G.split_scales && i < G.L    # don't split after last iteration
#             X, Z = tensor_split(X)
#             ΔX, ΔZx = tensor_split(ΔX)

#             X_save[i] = Z
#             ΔX_save[i] = ΔZx

#             G.Z_dims[i] = collect(size(X))
#         end
#     end

#     G.split_scales && (X = reshape(cat_states(X_save, X), orig_shape))
#     G.split_scales && (ΔX = reshape(cat_states(ΔX_save, ΔX), orig_shape))
#     return ΔX, X
# end


cat_scales(Yscales::Array{Any,1}) = cat(Yscales...; dims=1)

function uncat_scales(Y::AbstractArray{T,1}, dims::Array{NTuple{N,Int64},1}) where {T,N}
    Yscales = Array{Any,1}(undef,length(dims))
    i = 0
    for l = 1:length(dims)
        Yscales[l] = reshape(Y[i+1:i+prod(dims[l])], dims[l])
        i += prod(dims[l])
    end
    return Yscales
end

function clear_grad!(G::Glow)
    for l=1:G.nscales, k=1:G.depth
        clear_grad!(G.FS[k,l])
    end
end

function get_params(G::Glow)
    p = []
    for l=1:G.nscales, k=1:G.depth
        push!(p, get_params(G.FS[k,l]))
    end
    return cat(p...; dims=1)
end

function gpu(G::Glow{T,N}) where {T,N}
    FS = Array{FlowStep{T},2}(undef,G.depth,G.nscales)
    for l=1:G.nscales, k=1:G.depth
        FS[k,l] = gpu(G.FS[k,l])
    end
    return Glow{T,N}(G.depth, G.nscales, G.scale_dims, FS, G.logdet)
end

function cpu(G::Glow{T,N}) where {T,N}
    FS = Array{FlowStep{T},2}(undef,G.depth,G.nscales)
    for l=1:G.nscales, k=1:G.depth
        FS[k,l] = cpu(G.FS[k,l])
    end
    return Glow{T,N}(G.depth, G.nscales, G.scale_dims, FS, G.logdet)
end