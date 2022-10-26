export Conv1x1orth_fixed

struct Conv1x1orth_fixed{T<:Real} <: InvertibleNetwork
    nc::Int64
    P::Parameter#::AbstractMatrix{T}
    #P::AbstractMatrix{T}
    logdet::Bool
end

@Flux.functor Conv1x1orth_fixed

function Conv1x1orth_fixed(nc::Int64; logdet::Bool=true, T::DataType=Float32)

    P = Parameter(Array(qr(randn(T, nc, nc)).Q))
    #P = Array(qr(randn(T, nc, nc)).Q)
    return Conv1x1orth_fixed{T}(nc, P, logdet)

end

function forward(X::AbstractArray{T,N}, C::Conv1x1orth_fixed{T}) where {T,N}

    Y = conv1x1(X, C.P.data)
    C.logdet ? (return Y, T(0)) : (return Y)

end

inverse(Y::AbstractArray{T,N}, C::Conv1x1orth_fixed{T}) where {T,N} = return conv1x1(Y, toConcreteArray(C.P.data'))

backward(ΔY::AbstractArray{T,N}, Y::AbstractArray{T,N}, C::Conv1x1orth_fixed{T}) where {T,N} = inverse(ΔY, C), inverse(Y, C)
backward_inv(ΔY::AbstractArray{T,N}, Y::AbstractArray{T,N}, C::Conv1x1orth_fixed{T}) where {T,N} = forward(ΔY, C), forward(Y, C)


# Other utils
function clear_grad!(::Conv1x1orth_fixed) end

#get_params(C::Conv1x1orth_fixed{T}) where T = [Parameter([T(0)], [T(0)])]
get_params(C::Conv1x1orth_fixed{T}) where T = [Parameter(C.P.data, zero(C.P.data))]

gpu(C::Conv1x1orth_fixed{T}) where T = Conv1x1orth_fixed{T}(C.nc, gpu(C.P), C.logdet)
cpu(C::Conv1x1orth_fixed{T}) where T = Conv1x1orth_fixed{T}(C.nc, cpu(C.P), C.logdet)