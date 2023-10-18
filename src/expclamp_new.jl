export ExpClampLayerNew

ExpClampLayerNew(; clamp::Real=2) =
    InvertibleNetworks.ActivationFunction(
        X -> _expclamp_forward(X, clamp),
        Y -> _expclamp_inverse(Y, clamp),
        (ΔY, X) -> _expclamp_backward(ΔY, X, clamp))

_expclamp_forward(X::AbstractArray{T,N}, clamp::Real) where {T,N} = exp.(T(clamp) * T(0.636) * atan.(X))

_expclamp_inverse(Y::AbstractArray{T,N}, clamp::Real) where {T,N} = tan.(log.(Y)/T(clamp)/T(0.636))

function _expclamp_backward(ΔY::AbstractArray{T,N}, X::AbstractArray{T,N}, clamp::Real; Y=nothing) where {T,N}
    isnothing(Y) && (Y = _expclamp_forward(X, clamp))
    T(clamp)* T(0.636)*ΔY.*Y./(1 .+X.^2)
end