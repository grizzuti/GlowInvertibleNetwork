export ExpClampNewLayer, ExpClampNew, ExpClampNewGrad, SigmoidNewLayer, SigmoidNew, SigmoidNewGrad

ExpClampNew(X::AbstractArray{T,N}; α::T=T(1.9)) where{T,N} = exp.(T(2)*α/T(pi)*atan.(X/α))
function ExpClampNewGrad(ΔY::AbstractArray{T,N}, X::AbstractArray{T,N}; α::T=T(1.9)) where{T,N}
    Y = ExpClampNew(X; α=α)
    return T(2)/T(pi)*ΔY.*Y./(T(1).+(X/α).^2)
end

# SigmoidNew(X::AbstractArray{T,N}; α::T=T(0.5)) where{T,N} = (T(2)-α)*Sigmoid(X)+α*Sigmoid(-X)
SigmoidNew(X::AbstractArray{T,N}; α::T=T(0.5)) where{T,N} = Sigmoid(X)+α*Sigmoid(-X)
# SigmoidNewGrad(ΔY::AbstractArray{T,N}, X::AbstractArray{T,N}; α::T=T(0.5)) where {T,N} = (T(2)-α)*SigmoidGrad(ΔY, nothing; x=X)-α*SigmoidGrad(ΔY, nothing; x=-X)
SigmoidNewGrad(ΔY::AbstractArray{T,N}, X::AbstractArray{T,N}; α::T=T(0.5)) where {T,N} = SigmoidGrad(ΔY, nothing; x=X)-α*SigmoidGrad(ΔY, nothing; x=-X)

ExpClampNewLayer(α::T) where T = InvertibleNetworks.ActivationFunction(X->ExpClampNew(X;α=α), nothing, (ΔY,X)->ExpClampNewGrad(ΔY,X;α=α))

SigmoidNewLayer(α::T) where T = InvertibleNetworks.ActivationFunction(X->SigmoidNew(X;α=α), nothing, (ΔY,X)->SigmoidNewGrad(ΔY,X;α=α))