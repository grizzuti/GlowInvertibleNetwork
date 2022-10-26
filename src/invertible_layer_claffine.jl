export CouplingLayerAffine, CouplingLayerAffineOptions

struct CouplingLayerAffine{T<:Real} <: NeuralNetLayer
    CB::ConvolutionalBlock{T}
    affine::Bool
    activation::InvertibleNetworks.ActivationFunction
    logdet::Bool
end

@Flux.functor CouplingLayerAffine

struct CouplingLayerAffineOptions{T<:Real}
    options_convblock::ConvolutionalBlockOptions{T}
    activation::Union{Nothing,InvertibleNetworks.ActivationFunction}
    affine::Bool
end

CouplingLayerAffineOptions(; options_convblock::ConvolutionalBlockOptions{T}=ConvolutionalBlockOptions(), activation::Union{Nothing,InvertibleNetworks.ActivationFunction}=SigmoidNewLayer(T(0.5)), affine::Bool=true) where T = CouplingLayerAffineOptions{T}(options_convblock, activation, affine)

function CouplingLayerAffine(nc_in::Int64, nc_hidden::Int64;ndims=2, logdet::Bool=true, opt::CouplingLayerAffineOptions{T}=CouplingLayerAffineOptions()) where T

    mod(nc_in, 2) != 0 && throw(ErrorException("Number of channel dimension should be even"))
    nc_in_ = Int64(nc_in/2)
    nc_out = opt.affine ? nc_in : nc_in_
    CB = ConvolutionalBlock(nc_in_, nc_out, nc_hidden;ndims=ndims, opt=opt.options_convblock)
    return CouplingLayerAffine{T}(CB, opt.affine, opt.activation, logdet)

end

function forward(X::AbstractArray{T,N}, CL::CouplingLayerAffine{T}) where {T,N}

    X1, X2 = tensor_split(X)
    Y1 = X1
    t = CL.CB.forward(X1)
    if CL.affine
        logs, t = tensor_split(t)
        s = CL.activation.forward(logs)
        Y2 = X2.*s+t
        CL.logdet && (lgdt = logdet(CL, s))
    else
        Y2 = X2+t
        CL.logdet && (lgdt = T(0))
    end
    Y = tensor_cat(Y1, Y2)
    CL.logdet ? (return Y, lgdt) : (return Y)

end

function inverse(Y::AbstractArray{T,N}, CL::CouplingLayerAffine{T}) where {T,N}

    Y1, Y2 = tensor_split(Y)
    X1 = Y1
    t = CL.CB.forward(X1)
    if CL.affine
        logs, t = tensor_split(t)
        s = CL.activation.forward(logs)
        X2 = (Y2-t)./s
    else
        X2 = Y2-t
    end
    return tensor_cat(X1, X2)

end

function backward(ΔY::AbstractArray{T,N}, Y::AbstractArray{T,N}, CL::CouplingLayerAffine{T}) where {T,N}

    ΔY1, ΔY2 = tensor_split(ΔY); Y1, Y2 = tensor_split(Y)
    ΔX1 = ΔY1; X1 = Y1
    t = CL.CB.forward(X1)
    Δt = ΔY2
    if CL.affine
        logs, t = tensor_split(t)
        s = CL.activation.forward(logs)
        X2 = (Y2-t)./s
        ΔX2 = ΔY2.*s
        Δs = X2.*ΔY2
        CL.logdet && (Δs .-= dlogdet(CL, s))
        Δlogs = CL.activation.backward(Δs, logs)
        ΔX1 .+= CL.CB.backward(tensor_cat(Δlogs, Δt), X1)
    else
        X2 = Y2-t
        ΔX2 = ΔY2
        ΔX1 .+= CL.CB.backward(Δt, X1)
    end

    return tensor_cat(ΔX1, ΔX2), tensor_cat(X1, X2)
end

# function backward_inv(ΔX::AbstractArray{T, N}, X::AbstractArray{T, N}, L::CouplingLayerGlow; set_grad::Bool=true) where {T, N}

#     ΔX, X = L.C.forward((ΔX, X))
#     X1, X2 = tensor_split(X)
#     ΔX1, ΔX2 = tensor_split(ΔX)

#     # Recompute forward state
#     #Y, Y1, X2, S = forward(X, L; save=true)
#     logS_T = L.RB.forward(X2)
#     logSm, Tm = tensor_split(logS_T)
#     Sm = L.activation.forward(logSm)
#     Y1 = Sm.*X1 + Tm

#     # Backpropagate residual
#     ΔT = -ΔX1 ./ Sm
#     ΔS =  X1 .* ΔT 

#     ΔY2 = L.RB.backward(tensor_cat(L.activation.backward(ΔS, Sm), ΔT), X2) + ΔX2
#     ΔY1 = -ΔT

#     ΔY = tensor_cat(ΔY1, ΔY2)
#     Y  = tensor_cat(Y1, X2)

#     return ΔY, Y
# end

function backward_inv(ΔX::AbstractArray{T,N}, X::AbstractArray{T,N}, CL::CouplingLayerAffine{T}) where {T,N}

    ΔX1, ΔX2 = tensor_split(ΔX); X1, X2 = tensor_split(X)
    ΔY1 = ΔX1; Y1 = X1
    t = CL.CB.forward(Y1)
    
    logs, t = tensor_split(t)
    s = CL.activation.forward(logs)
    Y2 = s .* X2 + t

    Δt = -ΔX2 ./ s
    ΔY2 = - Δt
    Δs = Y2 .* Δt

    Δlogs = CL.activation.backward(Δs, logs)
    ΔY1 .+= CL.CB.backward(tensor_cat(Δlogs, Δt), Y1)
   
    return tensor_cat(ΔY1, ΔY2), tensor_cat(Y1, Y2)
end

## Other utils

clear_grad!(CL::CouplingLayerAffine) = clear_grad!(CL.CB)

get_params(CL::CouplingLayerAffine) = get_params(CL.CB)

 logdet(::CouplingLayerAffine{T}, s::AbstractArray{T,N}) where {T,N} = sum(log.(abs.(s)))/size(s,N)
dlogdet(::CouplingLayerAffine{T}, s::AbstractArray{T,N}) where {T,N} = T(1)./(s*size(s,N))

gpu(CL::CouplingLayerAffine{T}) where T = CouplingLayerAffine{T}(gpu(CL.CB), CL.affine, CL.activation, CL.logdet)
cpu(CL::CouplingLayerAffine{T}) where T = CouplingLayerAffine{T}(cpu(CL.CB), CL.affine, CL.activation, CL.logdet)