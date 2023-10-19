export Glow

mutable struct Glow <: InvertibleNetwork
    depth::Integer
    scales::Integer
    FS::Matrix{FlowStep}
    initial_squeeze::Bool
    logdet::Bool
    is_reversed::Bool
end

@Flux.functor Glow

function Glow(nc::Integer;
                nc_hidden::Integer=nc,
                stencil_size::NTuple{3,Integer}=(3,1,3), padding::NTuple{3,Integer}=(1,0,1), stride::NTuple{3,Integer}=(1,1,1),
                do_actnorm::Bool=true,
                activation::Union{Nothing,InvertibleNetworks.ActivationFunction}=ExpClampLayerNew(; clamp=2),
                init_id_an::Bool=false, init_id_q::Bool=false, init_id_cl::Bool=true,
                depth::Integer, scales::Integer,
                logdet::Bool=true,
                initial_squeeze::Bool=true,
                ndims::Integer=2)

    FS = Matrix{FlowStep}(undef, depth, scales)
    for l = 1:scales
        ((l > 1) || initial_squeeze) && (nc *= 2^ndims)
        for k = 1:depth
            FS[k,l] = FlowStep(nc;
                                nc_hidden=nc_hidden,
                                stencil_size=stencil_size, padding=padding, stride=stride,
                                do_actnorm=do_actnorm,
                                activation=activation,
                                logdet=logdet,
                                init_id_an=init_id_an, init_id_q=init_id_q, init_id_cl=init_id_cl,
                                ndims=ndims)
        end
        nc = div(nc, 2)
    end
    return Glow(depth, scales, FS, initial_squeeze, logdet, false)

end

function InvertibleNetworks.forward(X::AbstractArray{T,N}, G::Glow; logdet=nothing) where {T,N}
    isnothing(logdet) && (logdet = (G.logdet && ~G.is_reversed))

    # Input shape
    input_shape = size(X)

    # Keeping track of intermediate scale outputs
    Yscales = Vector{AbstractArray{T,N}}(undef, G.scales)

    # Initialize logdet
    logdet && (lgdt = T(0))

    # Loop over scales
    @inbounds for l = 1:G.scales

        # Squeeze
        (l > 1 || G.initial_squeeze) && (X = squeeze(X; pattern="checkerboard"))

        # Loop over depth
        @inbounds for k = 1:G.depth
            logdet ? ((X, lgdt_kl) = G.FS[k,l].forward(X); lgdt += lgdt_kl) :
                      (X = G.FS[k,l].forward(X))
        end

        # Split
        (l < G.scales) ? ((X, Yl) = tensor_split(X)) : (Yl = X) # Don't split last scale
        Yscales[l] = Yl

    end

    # Concatenating scales
    Y = cat_scales(Yscales, input_shape)

    logdet ? (return (Y, lgdt)) : (return Y)

end

function InvertibleNetworks.inverse(Y::AbstractArray{T,N}, G::Glow; logdet=nothing) where {T,N}
    isnothing(logdet) && (logdet = (G.logdet && G.is_reversed))

    # Initialize output
    X = []

    # De-concatenating scales
    scale_dims = compute_scale_dims(size(Y), G.scales; initial_squeeze=G.initial_squeeze)
    Yscales = uncat_scales(Y, scale_dims)

    # Initialize logdet
    logdet && (lgdt = T(0))

    # Loop over scales
    @inbounds for l = G.scales:-1:1

        # Concatenation
        (l < G.scales) ? (X = tensor_cat(X, Yscales[l])) : (X = Yscales[l])

        # Loop over depth
        @inbounds for k = G.depth:-1:1
            logdet ? ((X, lgdt_kl) = G.FS[k,l].inverse(X; logdet=true); lgdt += lgdt_kl) :
                      (X = G.FS[k,l].inverse(X; logdet=false))
        end

        # Unsqueeze
        (l > 1 || G.initial_squeeze) && (X = unsqueeze(X; pattern="checkerboard"))

    end

    logdet ? (return (X, lgdt)) : (return X)

end

function InvertibleNetworks.backward(ΔY::AbstractArray{T,N}, Y::AbstractArray{T,N}, G::Glow; set_grad::Bool=true) where {T,N}

    # Initialize output
    ΔX = []; X = []

    # De-concatenating scales
    scale_dims = compute_scale_dims(size(Y), G.scales; initial_squeeze=G.initial_squeeze)
    ΔYscales = uncat_scales(ΔY, scale_dims)
    Yscales  = uncat_scales(Y,  scale_dims)

    # Loop over scales
    @inbounds for l = G.scales:-1:1

        # Concatenation
        (l < G.scales) ? (X = tensor_cat(X, Yscales[l]); ΔX = tensor_cat(ΔX, ΔYscales[l])) :
                         (X = Yscales[l]; ΔX = ΔYscales[l])

        # Loop over depth
        @inbounds for k = G.depth:-1:1
            ΔX, X = G.FS[k,l].backward(ΔX, X; set_grad=set_grad)
        end

        # Unsqueeze
        (l > 1 || G.initial_squeeze) && (X = unsqueeze(X; pattern="checkerboard"); ΔX = unsqueeze(ΔX; pattern="checkerboard"))

    end

    return ΔX, X

end

function InvertibleNetworks.backward_inv(ΔX::AbstractArray{T,N}, X::AbstractArray{T,N}, G::Glow; set_grad::Bool=true) where {T,N}

    # Original input shape
    input_shape = size(X)

    # Keeping track of intermediate scale outputs
    ΔYscales = Vector{AbstractArray{T,N}}(undef, G.scales)
    Yscales  = Vector{AbstractArray{T,N}}(undef, G.scales)

     # Loop over scales
    @inbounds for l = 1:G.scales

        # Squeeze
        (l > 1 || G.initial_squeeze) && (X  = squeeze(X;  pattern="checkerboard");
                                         ΔX = squeeze(ΔX; pattern="checkerboard"))

        # Loop over depth
        @inbounds for k = 1:G.depth
            ΔX, X = G.FS[k,l].backward_inv(ΔX, X; set_grad=set_grad)
        end

        # Split
        (l < G.scales) ? ((ΔX, ΔYl) = tensor_split(ΔX); (X, Yl) = tensor_split(X)) :
                          (ΔYl = ΔX; Yl = X)
        ΔYscales[l] = ΔYl
        Yscales[l]  = Yl

    end

    # Concatenating scales
    ΔY = cat_scales(ΔYscales, input_shape)
    Y  = cat_scales(Yscales,  input_shape)

    return ΔY, Y

end


# Concatenation utils

function compute_scale_dims(input_size::NTuple{N,Integer}, scales::Integer; initial_squeeze::Bool=false) where N
    ndims = N-2
    nx = input_size[1:ndims]
    nc = input_size[ndims+1]
    nb = input_size[ndims+2]
    scale_dims = Vector{NTuple{N,Integer}}(undef, scales)
    @inbounds for l = 1:scales
        ((l > 1) || initial_squeeze) && (nc *= 2^ndims;
                                         nx = div.(nx, 2))
        (l < scales) && (nc = div(nc, 2))
        scale_dims[l] = (nx..., nc, nb)
    end
    return scale_dims
end

cat_scales(Yscales::AbstractVector{AT}, sz::NTuple{N,Integer}) where {T,N,AT<:AbstractArray{T,N}} = reshape(cat(vec.(Yscales)...; dims=1), sz)

function uncat_scales(Y::AbstractArray{T,N}, scale_dims::Vector{NTuple{N,Integer}}) where {T,N}
    l = length(scale_dims)
    Yscales = Vector{AbstractArray{T}}(undef, l)
    i::Int = 0
    for l = 1:l
        Yscales[l] = reshape(Y[i+1:i+prod(scale_dims[l])], scale_dims[l])
        i += prod(scale_dims[l])
    end
    return Yscales
end


# Other utils

function InvertibleNetworks.tag_as_reversed!(G::Glow, tag::Bool)
    @inbounds for l = 1:G.scales, k = 1:G.depth
        InvertibleNetworks.tag_as_reversed!(G.FS[k,l], tag)
    end
    G.is_reversed = tag
    return G
end

function InvertibleNetworks.set_params!(G::Glow, θ::AbstractVector{<:Parameter})
    set_params!(get_params(G), θ)
    @inbounds for l = 1:G.scales, k = 1:G.depth
        G.FS[k,l].Q.stencil = nothing
    end
end