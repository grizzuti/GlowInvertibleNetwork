export ActNormNew, reset!

mutable struct ActNormNew <: InvertibleNetwork
    s::Parameter
    b::Parameter
    is_init::Bool
    logdet::Bool
    is_reversed::Bool
end

@Flux.functor ActNormNew

# Constructor: Initialize with nothing
function ActNormNew(nc; logdet::Bool=true, init_id::Bool=false)
    s = Parameter(nothing)
    b = Parameter(nothing)
    AN = ActNormNew(s, b, false, logdet, false)
    init_id && init_id!(nc, AN)
    return AN
end

function InvertibleNetworks.forward(X::AbstractArray{T,N}, AN::ActNormNew; logdet=nothing) where {T,N}
    isnothing(logdet) && (logdet = (AN.logdet && ~AN.is_reversed))

    ~is_init(AN) && init_fw!(X, AN)
    inds = inds_reshape(N)
    Y = X.*reshape(AN.s.data, inds...).+reshape(AN.b.data, inds...)

    logdet ? (return (Y, logdet_fw_an(size(X)[1:N-2], AN.s.data))) : (return Y)

end

is_init(AN::ActNormNew) = AN.is_init

function init_fw!(X::AbstractArray{T,N}, AN::ActNormNew) where {T,N}
    μ = vec(mean(X; dims=(1:N-2...,N)))
    σ = sqrt.(vec(var(X; dims=(1:N-2...,N))))
    AN.s.data = 1 ./σ
    AN.b.data = -μ./σ
    AN.is_init = true
end

function init_id!(nc::Integer, AN::ActNormNew)
    AN.s.data = ones(T, nc)
    AN.b.data = zeros(T, nc)
    AN.is_init = true
end

inds_reshape(N::Integer) = Tuple([(i == N-1) ? Colon() : 1 for i = 1:N])

function InvertibleNetworks.inverse(Y::AbstractArray{T,N}, AN::ActNormNew; logdet=nothing) where {T, N}
    isnothing(logdet) && (logdet = (AN.logdet && AN.is_reversed))

    inds = inds_reshape(N)
    X = (Y.-reshape(AN.b.data, inds...))./reshape(AN.s.data, inds...)

    logdet ? (return (X, -logdet_fw_an(size(Y)[1:N-2], AN.s.data))) : (return X)

end

function InvertibleNetworks.backward(ΔY::AbstractArray{T,N}, Y::AbstractArray{T,N}, AN::ActNormNew; set_grad::Bool=true) where {T,N}

    inds = inds_reshape(N); dims = (1:N-2...,N)
    X = AN.inverse(Y; logdet=false)
    ΔX = ΔY.*reshape(AN.s.data, inds...)
    Δs = sum(ΔY.*X, dims=dims)[inds...]
    AN.logdet && (Δs -= logdet_bw_an(size(Y)[1:N-2], AN.s.data))
    Δb = sum(ΔY, dims=dims)[inds...]
    if set_grad
        AN.s.grad = Δs
        AN.b.grad = Δb
    end
    return ΔX, X

end

function InvertibleNetworks.backward_inv(ΔX::AbstractArray{T,N}, X::AbstractArray{T,N}, AN::ActNormNew; set_grad::Bool=true) where {T,N}

    inds = inds_reshape(N); dims = (1:N-2...,N); n = size(X)[1:N-2]
    Y = AN.forward(X; logdet=false)
    ΔY = ΔX./reshape(AN.s.data, inds...)
    Δs = -sum(ΔX.*X./reshape(AN.s.data, inds...), dims=dims)[inds...]
    AN.logdet && (Δs += logdet_bw_an(n, AN.s.data))
    Δb = -sum(ΔX./reshape(AN.s.data, inds...), dims=dims)[inds...]
    if set_grad
        AN.s.grad = Δs
        AN.b.grad = Δb
    end
    return ΔY, Y

end


# Logdet utils

logdet_fw_an(n::NTuple{N,Integer}, s::AbstractVector{T}) where {T,N} = prod(n)*sum(log.(abs.(s)))
logdet_bw_an(n::NTuple{N,Integer}, s::AbstractVector{T}) where {T,N} = prod(n)./s


# Other utils

InvertibleNetworks.tag_as_reversed!(AN::ActNormNew, tag::Bool) = (AN.is_reversed = tag; return AN)