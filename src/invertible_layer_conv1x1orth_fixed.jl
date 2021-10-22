export Conv1x1orth_fixed

struct Conv1x1orth_fixed{T<:Real} <: InvertibleNetwork
    nc::Int64
    P::AbstractMatrix{T}
    logdet::Bool
end

@Flux.functor Conv1x1orth_fixed

function Conv1x1orth_fixed(nc::Int64; logdet::Bool=true, T::DataType=Float32)

    P = Array(qr(randn(T, nc, nc)).Q)
    return Conv1x1orth_fixed{T}(nc, P, logdet)

end

function forward(X::AbstractArray{T,4}, C::Conv1x1orth_fixed{T}) where T

    Y = conv1x1(X, C.P)
    C.logdet ? (return Y, T(0)) : (return Y)

end

inverse(Y::AbstractArray{T,4}, C::Conv1x1orth_fixed{T}) where T = return conv1x1(Y, toConcreteArray(C.P'))

backward(ΔY::AbstractArray{T,4}, Y::AbstractArray{T,4}, C::Conv1x1orth_fixed{T}) where T = inverse(ΔY, C), inverse(Y, C)


# Other utils

function clear_grad!(::Conv1x1orth_fixed) end

get_params(::Conv1x1orth_fixed{T}) where T = [Parameter([T(0)], [T(0)])]

gpu(C::Conv1x1orth_fixed{T}) where T = Conv1x1orth_fixed{T}(C.nc, gpu(C.P), C.logdet)
cpu(C::Conv1x1orth_fixed{T}) where T = Conv1x1orth_fixed{T}(C.nc, cpu(C.P), C.logdet)