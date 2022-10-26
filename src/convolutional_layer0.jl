export ConvolutionalLayer0

struct ConvolutionalLayer0{T<:Real} <: NeuralNetLayer
    CL::ConvolutionalLayer{T}
    logscale_factor::T
    logs::Parameter
end

@Flux.functor ConvolutionalLayer0

function ConvolutionalLayer0(nc_in, nc_out;ndims=2, k=3, p=1, s=1, logscale_factor::Real=3., weight_std::Real=0., T::DataType=Float32)
    one_set = Tuple(1 for i=1:ndims)
    CL = ConvolutionalLayer(nc_in, nc_out;ndims=ndims, k=k, p=p, s=s, bias=true, weight_std=weight_std, T=T)
    logs = Parameter(zeros(T, one_set..., nc_out, 1))

    return ConvolutionalLayer0{T}(CL, logscale_factor, logs)

end

forward(X::AbstractArray{T,N}, CL0::ConvolutionalLayer0{T}; save::Bool=false) where {T,N} = forward(X, CL0.CL).*exp.(CL0.logs.data*CL0.logscale_factor)

function backward(ΔZ::AbstractArray{T,N}, X::AbstractArray{T,N}, CL0::ConvolutionalLayer0{T}; Z::Union{Nothing,AbstractArray{T,N}}=nothing) where {T,N}

    if N == 5
        dims_sum=(1,2,3,5)
    else 
        dims_sum=(1,2,4)
    end

    Z === nothing && (Z = forward(X, CL0)) # recompute forward pass, if not provided
    CL0.logs.grad = CL0.logscale_factor*sum(Z.*ΔZ; dims=dims_sum)
    ΔY = ΔZ.*exp.(CL0.logs.data*CL0.logscale_factor)
    ΔX = backward(ΔY, X, CL0.CL)

    return ΔX

end

function clear_grad!(CL0::ConvolutionalLayer0)
    clear_grad!(CL0.CL)
    CL0.logs.grad = nothing
end

get_params(CL0::ConvolutionalLayer0) = cat(get_params(CL0.CL), CL0.logs; dims=1)

gpu(CL0::ConvolutionalLayer0{T}) where T = ConvolutionalLayer0{T}(gpu(CL0.CL), CL0.logscale_factor, gpu(CL0.logs))
cpu(CL0::ConvolutionalLayer0{T}) where T = ConvolutionalLayer0{T}(cpu(CL0.CL), CL0.logscale_factor, cpu(CL0.logs))