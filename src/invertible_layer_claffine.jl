export CouplingLayerAffine

mutable struct CouplingLayerAffine <: InvertibleNetwork
    CB::ConvolutionalBlock
    activation::InvertibleNetworks.ActivationFunction
    logdet::Bool
    is_reversed::Bool
end

@Flux.functor CouplingLayerAffine

function CouplingLayerAffine(nc::Integer;
                                nc_hidden::Integer=nc,
                                stencil_size::NTuple{3,Integer}=(3,1,3),
                                padding::NTuple{3,Integer}=(1,0,1),
                                stride::NTuple{3,Integer}=(1,1,1),
                                do_actnorm::Bool=true,
                                activation::Union{Nothing,InvertibleNetworks.ActivationFunction}=ExpClampLayerNew(; clamp=2),
                                logdet::Bool=true,
                                init_id::Bool=true,
                                ndims::Integer=2)

    mod(nc, 2) != 0 && throw(ArgumentError("The number of input channels must be even"))
    CB = ConvolutionalBlock(div(nc, 2), nc_hidden, nc; stencil_size=stencil_size, padding=padding, stride=stride, do_actnorm=do_actnorm, init_zero=init_id, ndims=ndims)
    return CouplingLayerAffine(CB, activation, logdet, false)

end

function InvertibleNetworks.forward(X::AbstractArray{T,N}, CL::CouplingLayerAffine; logdet::Union{Nothing,Bool}=nothing, save::Bool=false) where {T,N}
    isnothing(logdet) && (logdet = (CL.logdet && ~CL.is_reversed))

    X1, X2 = tensor_split(X)
    Y1 = X1
    logs, t = tensor_split(CL.CB.forward(X1))
    s = CL.activation.forward(logs)
    Y2 = X2.*s+t
    Y = tensor_cat(Y1, Y2)
    save ? (return (Y, X1, X2, logs, s)) : (logdet ? (return (Y, compute_logdet(s))) : (return Y))

end

function InvertibleNetworks.inverse(Y::AbstractArray{T,N}, CL::CouplingLayerAffine; logdet::Union{Nothing,Bool}=nothing, save::Bool=false) where {T,N}
    isnothing(logdet) && (logdet = (CL.logdet && CL.is_reversed))

    Y1, Y2 = tensor_split(Y)
    X1 = Y1
    logs, t = tensor_split(CL.CB.forward(X1))
    s = CL.activation.forward(logs)
    X2 = (Y2-t)./s
    X = tensor_cat(X1, X2)
    save ? (return (X, Y1, X2, logs, s)) : (logdet ? (return (X, -compute_logdet(s))) : (return X))

end

function InvertibleNetworks.backward(ΔY::AbstractArray{T,N}, Y::AbstractArray{T,N}, CL::CouplingLayerAffine; set_grad::Bool=true) where {T,N}

    X, Y1, X2, logs, s = CL.inverse(Y; save=true)

    ΔY1, ΔY2 = tensor_split(ΔY)
    ΔX2 = ΔY2.*s
    ΔX1 = ΔY1
    Δs = X2.*ΔY2
    CL.logdet && (Δs .-= compute_dlogdet(s))
    Δlogs = CL.activation.backward(Δs, logs)
    Δt = ΔY2
    ΔX1 .+= CL.CB.backward(tensor_cat(Δlogs, Δt), Y1; set_grad=set_grad)
    ΔX = tensor_cat(ΔX1, ΔX2)

    return ΔX, X

end

function InvertibleNetworks.backward_inv(ΔX::AbstractArray{T,N}, X::AbstractArray{T,N}, CL::CouplingLayerAffine; set_grad::Bool=true) where {T,N}

    # Recompute inverse state
    Y, X1, X2, logs, s = CL.forward(X; save=true)

    # Backpropagate residual
    ΔX1, ΔX2 = tensor_split(ΔX)
    Δt = -ΔX2./s
    Δs = X2.*Δt
    Δs += compute_dlogdet(s)
    ΔY1 = CL.CB.backward(tensor_cat(CL.activation.backward(Δs, logs), Δt), X1; set_grad=set_grad)+ΔX1
    ΔY2 = -Δt
    ΔY = tensor_cat(ΔY1, ΔY2)

    return ΔY, Y

end


## Other utils

InvertibleNetworks.get_params(CL::CouplingLayerAffine) = get_params(CL.CB)

compute_logdet(s::AbstractArray{T,N}) where {T,N} = sum(log.(abs.(s)))/size(s,N)
compute_dlogdet(s::AbstractArray{T,N}) where {T,N} = 1 ./(s*size(s,N))

InvertibleNetworks.tag_as_reversed!(CL::CouplingLayerAffine, tag::Bool) = (CL.is_reversed = tag; return CL)