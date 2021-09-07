export FlowStep

struct FlowStep{T} <: NeuralNetLayer
    AN::ActNormPar{T}
    C::Conv1x1gen{T}
    CL::CouplingLayerAffine{T}
    logdet::Bool
end

@Flux.functor FlowStep

function FlowStep(nc, nc_hidden; logdet::Bool=true, T::DataType=Float32, cl_id::Bool=true)

    AN = ActNormPar(nc; logdet=logdet, T=T)
    C  = Conv1x1gen(nc; logdet=logdet, T=T)
    CL = CouplingLayerAffine(nc, nc_hidden; logdet=logdet, T=T, init_id=cl_id)
    return FlowStep{T}(AN,C,CL,logdet)

end

function forward(X::AbstractArray{T,4}, LG::FlowStep{T}) where T

    if LG.logdet
        X, logdet1 = LG.AN.forward(X)
        X, logdet2 = LG.C.forward(X)
        X, logdet3 = LG.CL.forward(X)
        return X, logdet1+logdet2+logdet3
    else
        X = LG.AN.forward(X)
        X = LG.C.forward(X)
        X = LG.CL.forward(X)
        return X
    end

end

function inverse(Y::AbstractArray{T,4}, LG::FlowStep{T}) where T

    Y = LG.CL.inverse(Y)
    Y = LG.C.inverse(Y)
    Y = LG.AN.inverse(Y)
    return Y

end

function backward(ΔY::AbstractArray{T,4}, Y::AbstractArray{T,4}, LG::FlowStep{T}) where T

    ΔY,Y = LG.CL.backward(ΔY,Y)
    ΔY,Y = LG.C.backward(ΔY,Y)
    ΔY,Y = LG.AN.backward(ΔY,Y)
    return ΔY,Y

end

function clear_grad!(LG::FlowStep)
    clear_grad!(LG.AN)
    clear_grad!(LG.C)
    clear_grad!(LG.CL)
end

get_params(LG::FlowStep) = cat(get_params(LG.AN), get_params(LG.C), get_params(LG.CL); dims=1)

gpu(LG::FlowStep{T}) where T = FlowStep{T}(gpu(LG.AN), gpu(LG.C), gpu(LG.CL), LG.logdet)
cpu(LG::FlowStep{T}) where T = FlowStep{T}(cpu(LG.AN), cpu(LG.C), cpu(LG.CL), LG.logdet)