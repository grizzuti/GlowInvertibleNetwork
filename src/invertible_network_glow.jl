export Glow

struct Glow{T} <: InvertibleNetwork
    depth::Int64
    nscales::Int64
    scale_dims::Array{NTuple{4,Int64},1}
    FS::AbstractArray{FlowStep{T},2}
    logdet::Bool
end

@Flux.functor Glow

function Glow(nc::Int64, nc_hidden::Int64, depth::Int64, nscales::Int64; logdet::Bool=true, cl_affine::Bool=true, cl_id::Bool=true, conv_id::Bool=true, conv_orth::Bool=false, T::DataType=Float32)

    FS = Array{FlowStep{T},2}(undef,depth,nscales)
    nc = 4*nc
    for l = 1:nscales
        for k = 1:depth
            FS[k,l] = FlowStep(nc, nc_hidden; logdet=logdet, cl_id=cl_id, cl_affine=cl_affine, conv_orth=conv_orth, conv_id=conv_id, T=T)
        end
        nc = 2*nc
    end
    return Glow{T}(depth,nscales,Array{NTuple{4,Int64}}(undef,nscales),FS,logdet)

end

function forward(X::AbstractArray{T,4}, G::Glow{T}) where T

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

function inverse(Y::AbstractArray{T,4}, G::Glow{T}) where T

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

function backward(ΔY::AbstractArray{T,4}, Y::AbstractArray{T,4}, G::Glow{T}) where T

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

cat_scales(Yscales::Array{Any,1}) = cat(Yscales...; dims=1)

function uncat_scales(Y::AbstractArray{T,1}, dims::Array{NTuple{4,Int64},1}) where T
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

function gpu(G::Glow{T}) where T
    FS = Array{FlowStep{T},2}(undef,G.depth,G.nscales)
    for l=1:G.nscales, k=1:G.depth
        FS[k,l] = gpu(G.FS[k,l])
    end
    return Glow{T}(G.depth, G.nscales, G.scale_dims, FS, G.logdet)
end

function cpu(G::Glow{T}) where T
    FS = Array{FlowStep{T},2}(undef,G.depth,G.nscales)
    for l=1:G.nscales, k=1:G.depth
        FS[k,l] = cpu(G.FS[k,l])
    end
    return Glow{T}(G.depth, G.nscales, G.scale_dims, FS, G.logdet)
end