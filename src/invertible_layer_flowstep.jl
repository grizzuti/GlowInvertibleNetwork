export FlowStep

mutable struct FlowStep <: InvertibleNetwork
    AN::ActNorm
    Q::OrthogonalConv1x1
    CL::CouplingLayerAffine
    logdet::Bool
    is_reversed::Bool
end

@Flux.functor FlowStep

function FlowStep(nc::Integer;
                    nc_hidden::Integer=nc,
                    stencil_size::NTuple{3,Integer}=(3,1,3),
                    padding::NTuple{3,Integer}=(1,0,1),
                    stride::NTuple{3,Integer}=(1,1,1),
                    do_actnorm::Bool=true,
                    activation::Union{Nothing,InvertibleNetworks.ActivationFunction}=SigmoidLayerNew(; low=0.5f0, high=1f0),
                    logdet::Bool=true,
                    init_id_an::Bool=false,
                    init_id_q::Bool=false,
                    init_id_cl::Bool=true,
                    ndims::Integer=2)

    AN = ActNorm(nc; logdet=logdet)
    Q  = OrthogonalConv1x1(nc; logdet=logdet, id_init=id_init_q)
    CL = CouplingLayerAffine(nc; nc_hidden=nc_hidden, stencil_size=stencil_size, padding=padding, stride=stride, do_actnorm=do_actnorm, activation=activation, logdet=logdet, init_id=init_id_cl, ndims=ndims)
    return FlowStep(AN, Q, CL, logdet, false)

end

function InvertibleNetworks.forward(X::AbstractArray{T,N}, FS::FlowStep; logdet::Union{Nothing,Bool}=nothing) where {T,N}
    isnothing(logdet) && (logdet = (FS.logdet && ~FS.is_reversed))

    logdet ? ((X, logdet1) = FS.AN.forward(X)) : (X = FS.AN.forward(X))
    logdet ? ((X, logdet2) = FS.Q.forward(X))  : (X = FS.Q.forward(X))
    logdet ? ((X, logdet3) = FS.CL.forward(X)) : (X = FS.CL.forward(X))
    logdet ? (return (X, logdet1+logdet2+logdet3)) : (return X)

end

function InvertibleNetworks.inverse(Y::AbstractArray{T,N}, FS::FlowStep; logdet::Union{Nothing,Bool}=nothing) where {T,N}
    isnothing(logdet) && (logdet = (FS.logdet && FS.is_reversed))

    logdet ? ((Y, logdet1) = FS.CL.inverse(Y)) : (Y = FS.CL.inverse(Y))
    logdet ? ((Y, logdet2) = FS.Q.inverse(Y))  : (Y = FS.Q.inverse(Y))
    logdet ? ((Y, logdet3) = FS.AN.inverse(Y)) : (Y = FS.AN.inverse(Y))
    logdet ? (return (Y, logdet1+logdet2+logdet3)) : (return Y)

end

function InvertibleNetworks.backward(ΔY::AbstractArray{T,N}, Y::AbstractArray{T,N}, FS::FlowStep; set_grad::Bool=true) where {T,N}

    ΔY, Y = FS.CL.backward(ΔY, Y; set_grad=set_grad)
    ΔY, Y = FS.Q.backward( ΔY, Y; set_grad=set_grad)
    ΔY, Y = FS.AN.backward(ΔY, Y; set_grad=set_grad)
    return ΔY,Y

end

function InvertibleNetworks.backward_inv(ΔY::AbstractArray{T,N}, Y::AbstractArray{T,N}, FS::FlowStep; set_grad::Bool=true) where {T,N}

    ΔY, Y = FS.CL.backward(ΔY, Y; set_grad=set_grad)
    ΔY, Y = FS.Q.backward( ΔY, Y; set_grad=set_grad)
    ΔY, Y = FS.AN.backward(ΔY, Y; set_grad=set_grad)
    return ΔY, Y

end

InvertibleNetworks.get_params(FS::FlowStep) = cat(get_params(FS.AN), get_params(FS.Q), get_params(FS.CL); dims=1)
InvertibleNetworks.tag_as_reversed!(FS::FlowStep, tag::Bool) = (FS.is_reversed = tag; return FS)