export ConvolutionalLayer0

struct ConvolutionalLayer0 <: InvertibleNetworks.NeuralNetwork
    CL::ConvolutionalLayer
    logscale_factor::Real
    logs::Parameter
end

@Flux.functor ConvolutionalLayer0

function ConvolutionalLayer0(nc_in, nc_out; stencil_size::Integer=3, padding::Integer=1, stride::Integer=1, bias::Bool=true, logscale_factor::Real=3., weight_std::Real=0., ndims::Integer=2)

    CL = ConvolutionalLayer(nc_in, nc_out; stencil_size=stencil_size, padding=padding, stride=stride, bias=bias, weight_std=weight_std, ndims=ndims)
    logs = Parameter(zeros(Float32, ones(Int, ndims)..., nc_out, 1))
    return ConvolutionalLayer0(CL, logscale_factor, logs)

end

InvertibleNetworks.forward(X::AbstractArray{T,N}, CL0::ConvolutionalLayer0) where {T,N} = CL0.CL.forward(X).*exp.(CL0.logs.data*T(CL0.logscale_factor))

function InvertibleNetworks.backward(ΔZ::AbstractArray{T,N}, X::AbstractArray{T,N}, CL0::ConvolutionalLayer0; Z::Union{Nothing,AbstractArray{T,N}}=nothing, set_grad::Bool=true) where {T,N}

    isnothing(Z) && (Z = CL0.forward(X)) # recompute forward pass, if not provided
    if set_grad
        isnothing(CL0.logs.grad) ? (CL0.logs.grad = CL0.logscale_factor*sum(Z.*ΔZ; dims=(1:N-2...,N))) : (CL0.logs.grad .= CL0.logscale_factor*sum(Z.*ΔZ; dims=(1:N-2...,N)))
    end
    ΔY = ΔZ.*exp.(CL0.logs.data*T(CL0.logscale_factor))
    ΔX = CL0.CL.backward(ΔY, X; set_grad=set_grad)
    return ΔX

end

InvertibleNetworks.get_params(CL0::ConvolutionalLayer0) = cat(get_params(CL0.CL), CL0.logs; dims=1)