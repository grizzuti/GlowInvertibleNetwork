export SigmoidLayerNew

SigmoidLayerNew(; low::Real=0f0, high::Real=1f0) =
    InvertibleNetworks.ActivationFunction(
        X -> _sigmoid_forward.(X, eltype(X)(low), eltype(X)(high)),
        Y -> _sigmoid_inverse.(Y, eltype(Y)(low), eltype(Y)(high)),
        (ΔY, Y; X=nothing) -> isnothing(X) ? _sigmoid_grad_y.(ΔY, Y, eltype(ΔY)(low), eltype(ΔY)(high)) : _sigmoid_grad_x.(ΔY, X, eltype(ΔY)(low), eltype(ΔY)(high)))

function _sigmoid_forward(x::T, low::T, high::T) where T
    t = exp(-abs(x))
    return ifelse(x ≥ 0, (low*t+high)/(1+t), (low+high*t)/(1+t))
end

function _sigmoid_inverse(y::T, low::T, high::T) where T
    (y >= (low+high)/2) ? (signx =  1; t = -(y-low)/(y-high)) : (signx = -1; t = -(y-high)/(y-low))
    return signx*log(t)
end

function _sigmoid_grad_y(Δy::T, y::T, low::T, high::T) where T
    x = _sigmoid_inverse(y, low, high)
    return _sigmoid_grad_x(Δy, x, low, high)
end

function _sigmoid_grad_x(Δy::T, x::T, low::T, high::T) where T
    t = exp(-abs(x))
    return Δy*(high-low)*t/(1+t)^2
end