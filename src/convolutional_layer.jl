export ConvolutionalLayer

mutable struct ConvolutionalLayer<:InvertibleNetworks.NeuralNetwork
    W::Parameter
    b::Union{Parameter,Nothing}
    padding::Integer
    stride::Integer
end

@Flux.functor ConvolutionalLayer

function ConvolutionalLayer(nc_in::Integer, nc_out::Integer; stencil_size::Integer=3, padding::Integer=1, stride::Integer=1, bias::Bool=true, weight_std::Real=0.05, ndims::Integer=2)

    W = Parameter(convert(Float32, weight_std)*randn(Float32, stencil_size*ones(Int, ndims)..., nc_in, nc_out))
    bias ? (b = Parameter(zeros(Float32, ones(Int, ndims)..., nc_out, 1))) : (b = nothing)
    return ConvolutionalLayer(W, b, padding, stride)

end

function InvertibleNetworks.forward(X::AbstractArray{T,N}, CL::ConvolutionalLayer) where {T,N}
    cdims = DenseConvDims(size(X), size(CL.W.data); padding=CL.padding, stride=CL.stride)
    Y = conv(X, CL.W.data, cdims)
    ~isnothing(CL.b) && (Y .+= CL.b.data)
    return Y
end

function InvertibleNetworks.backward(ΔY::AbstractArray{T,N}, X::AbstractArray{T,N}, CL::ConvolutionalLayer; set_grad::Bool=true) where {T,N}
    cdims = DenseConvDims(size(X), size(CL.W.data); padding=CL.padding, stride=CL.stride)
    ΔX = ∇conv_data(ΔY, CL.W.data, cdims)
    if set_grad
        isnothing(CL.W.grad) ? (CL.W.grad = ∇conv_filter(X, ΔY, cdims)) : (CL.W.grad .= ∇conv_filter(X, ΔY, cdims))
        if ~isnothing(CL.b)
            isnothing(CL.b.grad) ? (CL.b.grad = sum(ΔY, dims=(1:N-2...,N))) : (CL.b.grad .= sum(ΔY, dims=(1:N-2...,N)))
        end
    end
    return ΔX
end

InvertibleNetworks.get_params(CL::ConvolutionalLayer) = ~isnothing(CL.b) ? (return [CL.W, CL.b]) : (return [CL.W])