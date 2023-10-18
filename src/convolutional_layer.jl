export ConvolutionalLayer

abstract type ConvolutionalLayer<:InvertibleNetworks.NeuralNetwork end

mutable struct ConvolutionalLayerNoBias<:ConvolutionalLayer
    weight::Parameter
    padding::Integer
    stride::Integer
end

mutable struct ConvolutionalLayerWithBias<:ConvolutionalLayer
    CL::ConvolutionalLayerNoBias
    bias::Parameter
end

@Flux.functor ConvolutionalLayerNoBias
@Flux.functor ConvolutionalLayerWithBias

function ConvolutionalLayer(nc_in::Integer, nc_out::Integer; stencil_size::Integer=3, padding::Integer=1, stride::Integer=1, bias::Bool=true, init_zero::Bool=false, ndims::Integer=2)

    W = Parameter(~init_zero*glorot_uniform(stencil_size*ones(Int, ndims)..., nc_in, nc_out))
    C = ConvolutionalLayerNoBias(W, padding, stride)
    if bias
        b = Parameter(zeros(Float32, ones(Int, ndims)..., nc_out, 1))
        C = ConvolutionalLayerWithBias(C, b)
    end
    return C

end

weight(C::ConvolutionalLayerNoBias) = C.weight
weight(C::ConvolutionalLayerWithBias) = weight(C.CL)
has_bias(C::ConvolutionalLayerNoBias) = false
has_bias(C::ConvolutionalLayerWithBias) = true
bias(C::ConvolutionalLayerWithBias) = C.bias
padding(C::ConvolutionalLayerNoBias) = C.padding
padding(C::ConvolutionalLayerWithBias) = padding(C.CL)
stride(C::ConvolutionalLayerNoBias) = C.stride
stride(C::ConvolutionalLayerWithBias) = stride(C.CL)

function InvertibleNetworks.forward(X::AbstractArray{T,N}, CL::ConvolutionalLayer) where {T,N}
    W = weight(CL)
    cdims = DenseConvDims(size(X), size(W.data); padding=padding(CL), stride=stride(CL))
    Y = conv(X, W.data, cdims)
    has_bias(CL) && (Y .+= bias(CL).data)
    return Y
end

function InvertibleNetworks.backward(ΔY::AbstractArray{T,N}, X::AbstractArray{T,N}, CL::ConvolutionalLayer; set_grad::Bool=true) where {T,N}
    W = weight(CL)
    cdims = DenseConvDims(size(X), size(W.data); padding=padding(CL), stride=stride(CL))
    ΔX = ∇conv_data(ΔY, W.data, cdims)
    if set_grad
        W.grad = ∇conv_filter(X, ΔY, cdims)
        has_bias(CL) && (bias(CL).grad = sum(ΔY, dims=(1:N-2...,N)))
    end
    return ΔX
end