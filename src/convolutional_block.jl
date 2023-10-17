export ConvolutionalBlock

struct ConvolutionalBlock <: InvertibleNetworks.NeuralNetwork
    CL1::ConvolutionalLayer
    A1::Union{Nothing,ActNorm}
    CL2::ConvolutionalLayer
    A2::Union{Nothing,ActNorm}
    CL3::ConvolutionalLayer
    A3::Union{Nothing,ActNorm}
    init_zero::Bool
end

@Flux.functor ConvolutionalBlock

function ConvolutionalBlock(nc_in, nc_hidden, nc_out;
                                stencil_size::NTuple{3,Integer}=(3,1,3),
                                padding::NTuple{3,Integer}=(1,0,1),
                                stride::NTuple{3,Integer}=(1,1,1),
                                do_actnorm::Bool=true,
                                init_zero::Bool=true,
                                ndims::Integer=2)

    CL1 = ConvolutionalLayer(nc_in, nc_hidden; stencil_size=stencil_size[1], padding=padding[1], stride=stride[1], bias=~do_actnorm, ndims=ndims)
    do_actnorm ? (A1 = ActNorm(nc_hidden; logdet=false)) : (A1 = nothing)
    CL2 = ConvolutionalLayer(nc_hidden, nc_hidden; stencil_size=stencil_size[2], padding=padding[2], stride=stride[2], bias=~do_actnorm, ndims=ndims)
    do_actnorm ? (A2 = ActNorm(nc_hidden; logdet=false)) : (A2 = nothing)
    CL3 = ConvolutionalLayer(nc_hidden, nc_out; stencil_size=stencil_size[3], padding=padding[3], stride=stride[3], bias=~do_actnorm, init_zero=init_zero, ndims=ndims)
    do_actnorm ? (A3 = ActNorm(nc_hidden; logdet=false)) : (A3 = nothing)

    return ConvolutionalBlock(CL1, A1, CL2, A2, CL3, A3, init_zero)

end

function InvertibleNetworks.forward(X::AbstractArray{T,N}, CB::ConvolutionalBlock; save::Bool=false) where {N,T}

    C1 = CB.CL1.forward(X)
    ~isnothing(CB.A1) ? (A1 = CB.A1.forward(C1)) : (A1 = C1)
    H1 = ReLU(A1)

    C2 = CB.CL2.forward(H1)
    ~isnothing(CB.A2) ? (A2 = CB.A2.forward(C2)) : (A2 = C2)
    H2 = ReLU(A2)

    C3 = CB.CL3.forward(H2)
    if ~isnothing(CB.A3)
        if isnothing(CB.A3.s.data) && CB.init_zero
            CB.A3.s.data = similar(C3, size(C3, N-1)); fill!(CB.A3.s.data, 1)
            CB.A3.b.data = similar(C3, size(C3, N-1)); fill!(CB.A3.b.data, 0)
        end
        Y = CB.A3.forward(C3)
    else
        Y = C3
    end

    ~save ? (return Y) : (return C1, A1, H1, C2, A2, H2, C3, Y)

end

function InvertibleNetworks.backward(ΔY::AbstractArray{T, N}, X::AbstractArray{T,N}, CB::ConvolutionalBlock; set_grad::Bool=true) where {N,T}

    C1, A1, H1, C2, A2, H2, C3, Y = CB.forward(X; save=true) # Recompute forward states from input

    ~isnothing(CB.A3) ? (ΔC3 = custom_backward(ΔY, C3, CB.A3; set_grad=set_grad)) : (ΔC3 = ΔY)
    ΔH2 = CB.CL3.backward(ΔC3, H2; set_grad=set_grad)

    ΔA2 = ReLUgrad(ΔH2, A2)
    ~isnothing(CB.A2) ? (ΔC2 = custom_backward(ΔA2, C2, CB.A2; set_grad=set_grad)) : (ΔC2 = ΔA2)
    ΔH1 = CB.CL2.backward(ΔC2, H1; set_grad=set_grad)

    ΔA1 = ReLUgrad(ΔH1, A1)
    ~isnothing(CB.A1) ? (ΔC1 = custom_backward(ΔA1, C1, CB.A1; set_grad=set_grad)) : (ΔC1 = ΔA1)
    ΔX = CB.CL1.backward(ΔC1, X; set_grad=set_grad)

    return ΔX

end

function InvertibleNetworks.get_params(CB::ConvolutionalBlock)
    p = get_params(CB.CL1)
    ~isnothing(CB.A1) && (p = cat(p, get_params(CB.A1); dims=1))
    p = cat(p, get_params(CB.CL2); dims=1)
    ~isnothing(CB.A2) && (p = cat(p, get_params(CB.A2); dims=1))
    p = cat(p, get_params(CB.CL3); dims=1)
    ~isnothing(CB.A3) && (p = cat(p, get_params(CB.A3); dims=1))
    return p
end

function custom_backward(ΔY::AbstractArray{T,N}, X::AbstractArray{T,N}, AN::ActNorm; set_grad::Bool = true) where {T, N}
    inds = [i!=(N-1) ? 1 : Colon() for i=1:N]
    dims = collect(1:N-1); dims[end] +=1
    ΔX = ΔY .* reshape(AN.s.data, inds...)
    Δs = sum(ΔY .* X, dims=dims)[inds...]
    Δb = sum(ΔY, dims=dims)[inds...]
    if set_grad
        AN.s.grad = Δs
        AN.b.grad = Δb
    end
    return ΔX
end