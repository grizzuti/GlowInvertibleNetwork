export ExpClampNew, ExpClampNewGrad

ExpClampNew(X::AbstractArray{T,N}; α::T=T(1.9)) where{T,N} = exp.(T(2)*α/T(pi)*atan.(X/α))
function ExpClampNewGrad(ΔY::AbstractArray{T,N}, X::AbstractArray{T,N}; α::T=T(1.9)) where{T,N}
    Y = ExpClampNew(X; α=α)
    return T(2)/T(pi)*ΔY.*Y./(T(1).+(X/α).^2)
end