export Conv1x1gen

mutable struct Conv1x1gen{T<:Real} <: InvertibleNetwork
    nc::Int64
    W::AbstractMatrix{T}
    init_weight!::Bool
    P::AbstractMatrix{T}
    l::Parameter
    inds_l::AbstractVector
    u::Parameter
    inds_u::AbstractVector
    s::Parameter
    inds_s::AbstractVector
    logdet::Bool
end

@Flux.functor Conv1x1gen

function Conv1x1gen(nc::Int64; logdet::Bool=false, T::DataType=Float32)

    # Random orthogonal matrix
    W = Array(qr(randn(T, nc, nc)).Q)

    # LU decomposition PA=LU
    F = lu(W)
    P = F.P
    inds_l = findall((1:nc).>(1:nc)')
    L = F.L; l = Matrix2Array(L, inds_l)
    inds_u = findall((1:nc).<(1:nc)')
    U = F.U; u = Matrix2Array(U, inds_u)
    inds_s = findall((1:nc).==(1:nc)')
    s = abs.(diag(U)) # make sure W is SO(nc)

    return Conv1x1gen{T}(nc, similar(P), true, P, Parameter(l), inds_l, Parameter(u), inds_u, Parameter(s), inds_s, logdet)

end

function forward(X::AbstractArray{T,4}, C::Conv1x1gen{T}) where T

    if C.init_weight!
        C.W .= C.P'*(Array2Matrix(C.l.data, C.nc, C.inds_l)+idmat(X))*(Array2Matrix(C.u.data, C.nc, C.inds_u)+Array2Matrix(C.s.data, C.nc, C.inds_s))
        C.init_weight! = false
    end
    nx, ny, _, nb = size(X)
    Y = conv1x1(X, C.W)
    C.logdet ? (return Y, logdet(C, nx,ny,nb)) : (return Y)

end

function inverse(Y::AbstractArray{T,4}, C::Conv1x1gen{T}; Winv::Union{Nothing,AbstractMatrix{T}}=nothing) where T

    Winv === nothing && (Winv = C.W\idmat(Y))
    return conv1x1(Y, Winv)

end

function backward(ΔY::AbstractArray{T,4}, Y::AbstractArray{T,4}, C::Conv1x1gen{T}) where T

    # Backpropagating input
    ΔX = conv1x1(ΔY, toConcreteArray(C.W'))
    X = inverse(Y, C)

    # Backpropagating weights
    cdims = DenseConvDims(X, reshape(C.W, (1,1,size(C.W)...)); stride=(1,1), padding=(0,0))
    ΔW = reshape(∇conv_filter(X, ΔY, cdims), C.nc, C.nc)

    # Parameter gradient
    PΔW = C.P*ΔW
    LTPΔW = (Array2Matrix(C.l.data, C.nc, C.inds_l)+idmat(X))'*PΔW
    C.l.grad = Matrix2Array(PΔW*(Array2Matrix(C.u.data, C.nc, C.inds_u)+Array2Matrix(C.s.data, C.nc, C.inds_s))', C.inds_l)
    C.u.grad = Matrix2Array(LTPΔW, C.inds_u)
    C.s.grad = Matrix2Array(LTPΔW, C.inds_s)
    C.logdet && (C.s.grad .-= dlogdet(C, size(X,1),size(X,2),size(X,4)))

    # Resetting weight computation flag
    C.init_weight! = true

    return ΔX, X

end

# Log-det utils

logdet(C::Conv1x1gen, nx::Int64, ny::Int64, nb::Int64) = nx*ny*sum(log.(abs.(C.s.data)))/nb
dlogdet(C::Conv1x1gen, nx::Int64, ny::Int64, nb::Int64) = nx*ny./(C.s.data*nb)

# Convolutional weight utils

conv1x1(X::AbstractArray{T,4}, W::AbstractMatrix{T}) where T = conv(X, reshape(W, (1,1,size(W)...)); stride=(1,1), pad=(0,0))

idmat(X::Array{T}) where T = Matrix{T}(I,size(X,3),size(X,3))
idmat(X::CuArray{T}) where T = CuMatrix{T}(I,size(X,3),size(X,3))

toConcreteArray(X::Adjoint{T,Array{T,N}}) where {T,N} = Array(X)
toConcreteArray(X::Adjoint{T,CuArray{T,N,O}}) where {T,N,O} = CuArray(X)

# LU utils

function Array2Matrix(a::Array{T,1}, n::Int64, inds) where T
    A = zeros(T, n, n)
    A[inds] .= a
    return A
end

function Array2Matrix(a::CuArray{T,1}, n::Int64, inds) where T
    A = CUDA.zeros(T, n, n)
    A[inds] .= a
    return A
end

Matrix2Array(A::AbstractArray{T,2}, inds) where T = A[inds]

# Other utils

function clear_grad!(C::Conv1x1gen)
    C.l.grad = nothing
    C.u.grad = nothing
    C.s.grad = nothing
end

get_params(C::Conv1x1gen) = [C.l, C.u, C.s]

gpu(C::Conv1x1gen{T}) where T = Conv1x1gen{T}(C.nc, gpu(C.W), true, gpu(C.P), gpu(C.l), C.inds_l, gpu(C.u), C.inds_u, gpu(C.s), C.inds_s, C.logdet)
cpu(C::Conv1x1gen{T}) where T = Conv1x1gen{T}(C.nc, cpu(C.W), true, cpu(C.P), cpu(C.l), C.inds_l, cpu(C.u), C.inds_u, cpu(C.s), C.inds_s, C.logdet)