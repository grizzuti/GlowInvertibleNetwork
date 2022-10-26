export ConvolutionalLayer

struct ConvolutionalLayer{T<:Real} <: NeuralNetLayer
    W::Parameter
    b::Union{Parameter,Nothing}
    stride
    padding
end

@Flux.functor ConvolutionalLayer

function ConvolutionalLayer(nc_in, nc_out;ndims=2, k=3, p=1, s=1, bias::Bool=true, weight_std::Real=0.05, T::DataType=Float32)

    filter_k = Tuple(k for i=1:ndims)
    one_set = Tuple(1 for i=1:ndims)
    W = Parameter(T(weight_std)*randn(T, filter_k..., nc_in, nc_out))
    bias ? (b = Parameter(zeros(T, one_set..., nc_out, 1))) : (b = nothing)

    return ConvolutionalLayer{T}(W, b, s, p)
end

function forward(X::AbstractArray{T,N}, CL::ConvolutionalLayer{T}) where {T,N}
    # println(size(X))
    # println(size(CL.W.data))
    # println(typeof(X))
    # println(typeof(CL.W.data))
    # println(CL.stride)
    # println(CL.padding)
    Y = conv(X, CL.W.data; stride=CL.stride, pad=CL.padding)
    CL.b !== nothing && (Y .+= CL.b.data)
    return Y
end

function backward(ΔY::AbstractArray{T,N}, X::AbstractArray{T,N}, CL::ConvolutionalLayer{T}) where {T,N}

    if N == 5
        dims_sum=(1,2,3,5)
    else 
        dims_sum=(1,2,4)
    end

    cdims = DenseConvDims(X, CL.W.data; stride=CL.stride, padding=CL.padding)
    ΔX = ∇conv_data(ΔY, CL.W.data, cdims)
    CL.W.grad = ∇conv_filter(X, ΔY, cdims)
    CL.b !== nothing && (CL.b.grad = sum(ΔY, dims=dims_sum))

    return ΔX

end

function clear_grad!(CL::ConvolutionalLayer)
    CL.W.grad = nothing
    CL.b !== nothing && (CL.b.grad = nothing)
end

function get_params(CL::ConvolutionalLayer)
    CL.b !== nothing ? (return [CL.W, CL.b]) : (return [CL.W])
end

gpu(CL::ConvolutionalLayer{T}) where T = ConvolutionalLayer{T}(gpu(CL.W), gpu(CL.b), CL.stride, CL.padding)
cpu(CL::ConvolutionalLayer{T}) where T = ConvolutionalLayer{T}(cpu(CL.W), cpu(CL.b), CL.stride, CL.padding)