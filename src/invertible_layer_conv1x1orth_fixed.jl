export Conv1x1orth_fixed

struct Conv1x1orth_fixed{T<:Real} <: InvertibleNetwork
    nc::Int64
    P::AbstractMatrix{T}
    Pinv::AbstractMatrix{T}
    p::Parameter
    logdet::Bool
end

@Flux.functor Conv1x1orth_fixed

function Conv1x1orth_fixed(nc::Int64; logdet::Bool=true, T::DataType=Float32)

    P = Array(qr(randn(T, nc, nc)).Q)
    Pinv = P\idmat(T, nc)
    return Conv1x1orth_fixed{T}(nc, P, Pinv, Parameter([T(0)]), logdet)

end

function forward(X::AbstractArray{T,4}, C::Conv1x1orth_fixed{T}) where T

    Y = conv1x1(X, C.P)
    C.logdet ? (return Y, T(0)) : (return Y)

end

inverse(Y::AbstractArray{T,4}, C::Conv1x1orth_fixed{T}) where T = return conv1x1(Y, C.Pinv)

function backward(ΔY::AbstractArray{T,4}, Y::AbstractArray{T,4}, C::Conv1x1orth_fixed{T}) where T

    # Backpropagating input
    ΔX = conv1x1(ΔY, toConcreteArray(C.P'))
    X = inverse(Y, C)
    C.p.grad = [T(0)]
    return ΔX, X

end


# Other utils

function clear_grad!(::Conv1x1orth_fixed) end

get_params(C::Conv1x1orth_fixed{T}) where T = [C.p]

gpu(C::Conv1x1orth_fixed{T}) where T = Conv1x1orth_fixed{T}(C.nc, gpu(C.P), gpu(C.Pinv), gpu(C.p), C.logdet)
cpu(C::Conv1x1orth_fixed{T}) where T = Conv1x1orth_fixed{T}(C.nc, cpu(C.P), cpu(C.Pinv), cpu(C.p), C.logdet)