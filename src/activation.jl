export ExpClampNewLayer, ExpClampNew, ExpClampNewGrad, SigmoidNewLayer, SigmoidNew, SigmoidNewGrad

ExpClampNew(X::AbstractArray{T,N}; α::T=T(1.9)) where{T,N} = exp.(T(2)*α/T(pi)*atan.(X/α))
function ExpClampNewGrad(ΔY::AbstractArray{T,N}, X::AbstractArray{T,N}; α::T=T(1.9)) where{T,N}
    Y = ExpClampNew(X; α=α)
    return T(2)/T(pi)*ΔY.*Y./(T(1).+(X/α).^2)
end

SigmoidNew(X::AbstractArray{T,N}; α::T=T(0.5)) where{T,N} = (T(1)-α)*Sigmoid(X).+α
function SigmoidNewGrad(ΔY::AbstractArray{T,N}, X::AbstractArray{T,N}; α::T=T(0.5)) where{T,N}
    ΔX = SigmoidGrad(ΔY, nothing; x=X)
    return (T(1)-α)*ΔX
end

ExpClampNewLayer(α::T) where T = InvertibleNetworks.ActivationFunction(X->ExpClampNew(X;α=α), nothing, (ΔY,X)->ExpClampNewGrad(ΔY,X;α=α))

SigmoidNewLayer(α::T) where T = InvertibleNetworks.ActivationFunction(X->SigmoidNew(X;α=α), nothing, (ΔY,X)->SigmoidNewGrad(ΔY,X;α=α))