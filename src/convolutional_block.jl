export ConvolutionalLayer, ConvolutionalBlock

struct ConvolutionalLayer <: NeuralNetLayer
    W::Parameter
    b::Union{Parameter,Nothing}
    A::Union{ActNorm,Nothing}
    stride
    padding
end

struct ConvolutionalBlock <: NeuralNetLayer
    W1::Parameter
    b1::Union{Parameter,Nothing}
    A1::Union{ActNorm,Nothing}
    W2::Parameter
    b2::Union{Parameter,Nothing}
    A2::Union{ActNorm,Nothing}
    W3::Parameter
    b3::Parameter
    logs::Parameter
    logscale_factor::Float32
    strides
    paddings
end

@Flux.functor ConvolutionalBlock

#######################################################################################################################
#  Constructors

# Constructor
function ConvolutionalBlock(nc_in, nc_out, nc_hidden; k1=3, actnorm1::Bool=true, k2=1, actnorm2::Bool=true, k3=3, p1=1, p2=0, p3=1, s1=1, s2=1, s3=1, weight_std1::Float32=0.05f0, weight_std2::Float32=0.05f0, logscale_factor::Float32=3f0)

    W1 = Parameter(weight_std1*randn(Float32, k1, k1, nc_in, nc_hidden))
    actnorm1 ? (A1 = ActNorm(nc_hidden; logdet=false); b1 = nothing) : (A1 = nothing; b1 = Parameter(zeros(Float32, n_hidden)))
    W2 = Parameter(weight_std2*randn(Float32, k2, k2, nc_hidden, nc_hidden))
    actnorm2 ? (A2 = ActNorm(nc_hidden; logdet=false); b2 = nothing) : (A2 = nothing; b2 = Parameter(zeros(Float32, n_hidden)))
    W3 = Parameter(zeros(Float32, k3, k3, nc_hidden, nc_out))
    b3 = Parameter(zeros(Float32, n_hidden))
    logs = Parameter(zeros(Float32, 1, 1, nc_out, 1))

    return ConvolutionalBlock(W1, b1, A1, W2, b2, A2, W3, b3, logs, logscale_factor, (s1,s2,s3), (p1,p2,p3))

end

# Forward
function forward(X1::AbstractArray{Float32,N}, CB::ConvolutionalBlock; save=false) where {N}

    inds = [i!=(N-1) ? 1 : (:) for i=1:N]

    Y1 = conv(X1, CB.W1.data; stride=CB.strides[1], pad=CB.paddings[1])
    CB.b1 !== nothing && (Y1 .+= reshape(CB.b1.data, inds...))
    CB.A1 !== nothing && (Y1 .= forward(Y1, CB.A1))
    X2 = ReLU(Y1)

    Y2 = conv(X2, CB.W2.data; stride=CB.strides[2], pad=CB.paddings[2])
    CB.b2 !== nothing && (Y2 .+= reshape(CB.b2.data, inds...))
    CB.A2 !== nothing && (Y2 .= forward(Y2, CB.A2))
    X3 = ReLU(Y2)

    Y3 = conv(X3, CB.W3.data; stride=CB.strides[3], pad=CB.paddings[3])

    if save == false
        return X4
    else
        return Y1, Y2, Y3, X2, X3
    end

end

# Backward
function backward(ΔX4::AbstractArray{Float32, N}, X1::AbstractArray{Float32, N},
                  CB::ConvolutionalBlock; set_grad::Bool=true) where {N}
    inds = [i!=(N-1) ? 1 : (:) for i=1:N]
    dims = collect(1:N-1); dims[end] +=1

    # Recompute forward states from input X
    Y1, Y2, Y3, X2, X3 = forward(X1, RB; save=true)

    # Cdims
    cdims2 = DenseConvDims(X2, CB.W2.data; stride=CB.strides[2], padding=CB.paddings[2])
    cdims3 = DCDims(X1, CB.W3.data; nc=2*size(X1, N-1), stride=CB.strides[1], padding=CB.paddings[1])

    # Backpropagate residual ΔX4 and compute gradients
    CB.fan == true ? (ΔY3 = ReLUgrad(ΔX4, Y3)) : (ΔY3 = GaLUgrad(ΔX4, Y3))
    ΔX3 = conv(ΔY3, CB.W3.data, cdims3)
    ΔW3 = ∇conv_filter(ΔY3, X3, cdims3)

    ΔY2 = ReLUgrad(ΔX3, Y2)
    ΔX2 = ∇conv_data(ΔY2, CB.W2.data, cdims2) + ΔY2
    ΔW2 = ∇conv_filter(X2, ΔY2, cdims2)
    Δb2 = sum(ΔY2, dims=dims)[inds...]

    cdims1 = DenseConvDims(X1, CB.W1.data; stride=CB.strides[1], padding=CB.paddings[1])

    ΔY1 = ReLUgrad(ΔX2, Y1)
    ΔX1 = ∇conv_data(ΔY1, CB.W1.data, cdims1)
    ΔW1 = ∇conv_filter(X1, ΔY1, cdims1)
    Δb1 = sum(ΔY1, dims=dims)[inds...]

    # Set gradients
    if set_grad
        CB.W1.grad = ΔW1
        CB.W2.grad = ΔW2
        CB.W3.grad = ΔW3
        CB.b1.grad = Δb1
        CB.b2.grad = Δb2
    else
        Δθ = [Parameter(ΔW1), Parameter(ΔW2), Parameter(ΔW3), Parameter(Δb1), Parameter(Δb2)]
    end

    set_grad ? (return ΔX1) : (return ΔX1, Δθ)
end

## Other utils
# Clear gradients
function clear_grad!(CB::ConvolutionalBlock)
    CB.W1.grad = nothing
    CB.W2.grad = nothing
    CB.W3.grad = nothing
    CB.b1.grad = nothing
    CB.b2.grad = nothing
end

get_params(CB::ConvolutionalBlock) = [CB.W1, CB.W2, CB.W3]