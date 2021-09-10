export CouplingLayerAffine

struct CouplingLayerAffine{T<:Real} <: NeuralNetLayer
    CB::ConvolutionalBlock{T}
    affine::Bool
    logdet::Bool
end

@Flux.functor CouplingLayerAffine

function CouplingLayerAffine(nc_in::Int64, nc_hidden::Int64; logdet::Bool=true, init_id::Bool=true, affine::Bool=true, T::DataType=Float32)

    mod(nc_in, 2) != 0 && throw(ErrorException("Number of channel dimension should be even"))
    nc_in_ = Int64(nc_in/2)
    nc_out = affine ? nc_in : nc_in_
    CB = ConvolutionalBlock(nc_in_, nc_out, nc_hidden; T=T, init_zero=init_id)
    return CouplingLayerAffine{T}(CB, affine, logdet)

end

function forward(X::AbstractArray{T,4}, CL::CouplingLayerAffine{T}) where T

    X1, X2 = tensor_split(X)
    Y1 = X1
    t = CL.CB.forward(X1)
    if CL.affine
        logs, t = tensor_split(t)
        # s = 2*Sigmoid(logs)
        s = ExpClampNew(logs)
        Y2 = X2.*s+t
        CL.logdet && (lgdt = logdet(CL, s))
    else
        Y2 = X2+t
        CL.logdet && (lgdt = T(0))
    end
    CL.logdet ? (return tensor_cat(Y1, Y2), lgdt) : (return tensor_cat(Y1, Y2))

end

function inverse(Y::AbstractArray{T,4}, CL::CouplingLayerAffine{T}) where T

    Y1, Y2 = tensor_split(Y)
    X1 = Y1
    t = CL.CB.forward(X1)
    if CL.affine
        logs, t = tensor_split(t)
        # s = 2*Sigmoid(logs)
        s = ExpClampNew(logs)
        X2 = (Y2-t)./s
    else
        X2 = Y2-t
    end
    return tensor_cat(X1, X2)

end

function backward(ΔY::AbstractArray{T,4}, Y::AbstractArray{T,4}, CL::CouplingLayerAffine{T}) where T

    ΔY1, ΔY2 = tensor_split(ΔY); Y1, Y2 = tensor_split(Y)
    ΔX1 = ΔY1; X1 = Y1
    t = CL.CB.forward(X1)
    Δt = ΔY2
    if CL.affine
        logs, t = tensor_split(t)
        # s = 2*Sigmoid(logs)
        s = ExpClampNew(logs)
        X2 = (Y2-t)./s
        ΔX2 = ΔY2.*s
        Δs = X2.*ΔY2
        CL.logdet && (Δs .-= dlogdet(CL, s))
        # Δlogs = 2*SigmoidGrad(Δs, s; x=logs)
        Δlogs = ExpClampNewGrad(Δs, logs)
        ΔX1 .+= CL.CB.backward(tensor_cat(Δlogs, Δt), X1)
    else
        X2 = Y2-t
        ΔX2 = ΔY2
        ΔX1 .+= CL.CB.backward(Δt, X1)
    end

    return tensor_cat(ΔX1, ΔX2), tensor_cat(X1, X2)

end

## Other utils

clear_grad!(CL::CouplingLayerAffine) = clear_grad!(CL.CB)

get_params(CL::CouplingLayerAffine) = get_params(CL.CB)

 logdet(::CouplingLayerAffine{T}, s::AbstractArray{T,4}) where T = sum(log.(abs.(s)))/size(s,4)
dlogdet(::CouplingLayerAffine{T}, s::AbstractArray{T,4}) where T = T(1)./(s*size(s,4))

gpu(CL::CouplingLayerAffine{T}) where T = CouplingLayerAffine{T}(gpu(CL.CB), CL.affine, CL.logdet)
cpu(CL::CouplingLayerAffine{T}) where T = CouplingLayerAffine{T}(cpu(CL.CB), CL.affine, CL.logdet)