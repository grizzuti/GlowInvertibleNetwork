export OrthogonalConv1x1

mutable struct OrthogonalConv1x1 <: InvertibleNetwork

    # Stencil-related fields
    stencil_pars::Parameter
    pars2mat_idx
    nc::Integer
    stencil::Union{AbstractArray,Nothing}

    # Internal flags
    logdet::Bool
    is_reversed::Bool

    # Internal parameters related to the stencil exponential or derivative thereof
    log_mat::Union{AbstractArray,Nothing}
    niter_expder::Union{Nothing,Real}
    tol_expder::Union{Nothing,Real}

end

@Flux.functor OrthogonalConv1x1


# Constructor

function OrthogonalConv1x1(nc::Integer; logdet::Bool=true, id_init::Bool=false, niter_expder::Union{Nothing,Integer}=nothing, tol_expder::Union{Nothing,Real}=nothing)

    id_init ? (stencil_pars = vec2par(zeros(Float32, div(nc*(nc-1), 2)), (div(nc*(nc-1), 2), ))) :
              (stencil_pars = vec2par(glorot_uniform(div(nc*(nc-1), 2)), (div(nc*(nc-1), 2), )))
    pars2mat_idx = InvertibleNetworks._skew_symmetric_indices(nc)
    return OrthogonalConv1x1(stencil_pars, pars2mat_idx, nc, nothing,
                             logdet, false,
                             nothing, niter_expder, tol_expder)

end


# Forward/inverse/backward

function InvertibleNetworks.forward(X::AbstractArray{T,N}, C::OrthogonalConv1x1; logdet=nothing) where {T,N}
    isnothing(logdet) && (logdet = (C.logdet && ~C.is_reversed))

    # Compute exponential stencil
    isnothing(C.stencil) && _compute_exponential_stencil!(C, N-2; set_log=true)

    # Convolution
    cdims = DenseConvDims(size(X), size(C.stencil))
    Y = conv(X, C.stencil, cdims)

    return logdet ? (Y, convert(T, 0)) : Y

end

function InvertibleNetworks.inverse(Y::AbstractArray{T,N}, C::OrthogonalConv1x1; logdet=nothing) where {T,N}
    isnothing(logdet) && (logdet = (C.logdet && C.is_reversed))

    # Compute exponential stencil
    isnothing(C.stencil) && _compute_exponential_stencil!(C, N-2; set_log=true)

    # Convolution (adjoint)
    cdims = DenseConvDims(size(Y), size(C.stencil))
    X = ∇conv_data(Y, C.stencil, cdims)

    return logdet ? (X, convert(T, 0)) : X

end

function InvertibleNetworks.backward(ΔY::AbstractArray{T,N}, Y::AbstractArray{T,N}, C::OrthogonalConv1x1; set_grad::Bool=true) where {T,N}

    # Compute exponential stencil
    isnothing(C.stencil) && _compute_exponential_stencil!(C, N-2; set_log=true)

    # Convolution (adjoint)
    cdims = DenseConvDims(size(Y), size(C.stencil))
    X  = ∇conv_data(Y,  C.stencil, cdims)
    ΔX = ∇conv_data(ΔY, C.stencil, cdims)

    # Parameter gradient
    if set_grad
        Δstencil = reshape(∇conv_filter(X, ΔY, cdims), C.nc, C.nc)
        ΔA = InvertibleNetworks._Frechet_derivative_exponential(C.log_mat', Δstencil; niter=C.niter_expder, tol=isnothing(C.tol_expder) ? nothing : T(C.tol_expder))
        Δstencil_pars = ΔA[C.pars2mat_idx[1]]-ΔA[C.pars2mat_idx[2]]
        isnothing(C.stencil_pars.grad) ? (C.stencil_pars.grad = Δstencil_pars) : (C.stencil_pars.grad .= Δstencil_pars)
        C.stencil = nothing
    end

    return ΔX, X

end

function InvertibleNetworks.backward_inv(ΔX::AbstractArray{T,N}, X::AbstractArray{T,N}, C::OrthogonalConv1x1; set_grad::Bool=true) where {T,N}

    # Compute exponential stencil
    isnothing(C.stencil) && _compute_exponential_stencil!(C, N-2; set_log=true)

    # Convolution (adjoint)
    cdims = DenseConvDims(size(X), size(C.stencil))
    Y  = conv(X,  C.stencil, cdims)
    ΔY = conv(ΔX, C.stencil, cdims)

    # Parameter gradient
    if set_grad
        Δstencil = reshape(∇conv_filter(X, ΔY, cdims), C.nc, C.nc)
        ΔA = InvertibleNetworks._Frechet_derivative_exponential(C.log_mat', Δstencil; niter=C.niter_expder, tol=isnothing(C.tol_expder) ? nothing : T(C.tol_expder))
        Δstencil_pars = ΔA[C.pars2mat_idx[1]]-ΔA[C.pars2mat_idx[2]]
        isnothing(C.stencil_pars.grad) ? (C.stencil_pars.grad = -Δstencil_pars) : (C.stencil_pars.grad .= -Δstencil_pars)
        C.stencil = nothing
    end

    return ΔY, Y

end

function InvertibleNetworks.set_params!(C::OrthogonalConv1x1, θ::AbstractVector{<:Parameter})
    (length(θ) != 1) && throw(ArgumentError("Parameter not compatible"))
    C.stencil_pars = θ[1]; C.stencil = nothing
end


# Internal utilities for OrthogonalConv1x1

function _compute_exponential_stencil!(C::OrthogonalConv1x1, ndims::Integer; set_log::Bool=false)
    log_mat = InvertibleNetworks._pars2skewsymm(C.stencil_pars.data, C.pars2mat_idx, C.nc)
    C.stencil = _mat2stencil(InvertibleNetworks._exponential(log_mat), ndims)
    set_log && (C.log_mat = log_mat)
end

_mat2stencil(A::AbstractMatrix, ndims::Integer) = reshape(A, Tuple(ones(Int, ndims))..., size(A)...)


# Other

InvertibleNetworks.tag_as_reversed!(C::OrthogonalConv1x1, tag::Bool) = (C.is_reversed = tag; return C)