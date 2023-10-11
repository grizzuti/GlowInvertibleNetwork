function gradient_test_input(G, loss::Function, X::AbstractArray{T}; step::T=1f-4, rtol::T=1f-3, invnet::Bool=true) where T

    # Computing gradients
    hasfield(typeof(G), :logdet) ? (logdet = G.logdet) : (logdet = false)
    logdet ? ((Y, _) = G.forward(X)) : (Y = G.forward(X))
    _, ΔY = loss(Y)
    invnet ? ((ΔX, _) = G.backward(ΔY, Y)) : (ΔX = G.backward(ΔY, X))

    # Perturbations
    dX = typeof(X)(randn(T, size(X))); dX .*= norm(X)/norm(dX)

    # Test (wrt input)
    logdet ? ((Yp1, logdet_p1) = G.forward(X+T(0.5)*step*dX)) : (Yp1 = G.forward(X+T(0.5)*step*dX))
    lp1, _ = loss(Yp1)
    logdet && (lp1 -= logdet_p1)
    logdet ? ((Ym1, logdet_m1) = G.forward(X-T(0.5)*step*dX)) : (Ym1 = G.forward(X-T(0.5)*step*dX))
    lm1, _ = loss(Ym1)
    logdet && (lm1 -= logdet_m1)
    @test (lp1-lm1)/step ≈ dot(ΔX, dX) rtol=rtol

end

function gradient_test_pars(G, loss::Function, X::AbstractArray{T}; step::T=1e-4, rtol::T=1e-3, invnet::Bool=true, dθ::Union{Nothing,Array{Parameter,1}}=nothing) where T

    # Perturbations
    G = deepcopy(G); θ = deepcopy(get_params(G))
    if isnothing(dθ)
        dθ = Array{Parameter,1}(undef, length(θ))
        for i = eachindex(θ)
            dθ[i] = Parameter(typeof(θ[i].data)(randn(T, size(θ[i].data))))
            norm(θ[i].data) != T(0) && (dθ[i].data .*= norm(θ[i].data)/norm(dθ[i].data))
        end
    end

    # Computing gradients
    hasfield(typeof(G), :logdet) ? (logdet = G.logdet) : (logdet = false)
    logdet ? ((Y, _) = G.forward(X)) : (Y = G.forward(X))
    _, ΔY = loss(Y)
    invnet ? ((ΔX, _) = G.backward(ΔY, Y)) : (ΔX = G.backward(ΔY, X))
    Δθ = deepcopy(get_grads(G))

    # Test (wrt pars)
    set_params!(G, θ+T(0.5)*step*dθ)
    logdet ? ((Yp1, logdet_p1) = G.forward(X)) : (Yp1 = G.forward(X))
    lp1, _ = loss(Yp1)
    logdet && (lp1 -= logdet_p1)

    set_params!(G, θ-T(0.5)*step*dθ)
    logdet ? ((Ym1, logdet_m1) = G.forward(X)) : (Ym1 = G.forward(X))
    lm1, _ = loss(Ym1)
    logdet && (lm1 -= logdet_m1)
    @test (lp1-lm1)/step ≈ dot(Δθ, dθ) rtol=rtol

end

function cpu_vs_gpu_test(G, input_shape; rtol::Float32=1f-5, invnet::Bool=true)

    hasfield(typeof(G), :logdet) ? (logdet = G.logdet) : (logdet = false)
    Ggpu = gpu(deepcopy(G))
    Gcpu = cpu(deepcopy(G))

    # Forward
    Xcpu = randn(Float32, input_shape)
    Xgpu = gpu(deepcopy(Xcpu))
    # Ggpu isa Conv1x1gen && (Ggpu.init_weight! = true)
    # Gcpu isa Conv1x1gen && (Gcpu.init_weight! = true)
    if logdet
        Ygpu, logdet_gpu = Ggpu.forward(Xgpu)
        Ycpu, logdet_cpu = Gcpu.forward(Xcpu)
    else
        Ygpu = Ggpu.forward(Xgpu)
        Ycpu = Gcpu.forward(Xcpu)
    end
    Ygpu = cpu(Ygpu)
    out_shape = size(Ygpu)
    @test Ygpu ≈ Ycpu rtol=rtol
    logdet && (@test logdet_gpu ≈ logdet_cpu rtol=1f1*rtol)

    # Inverse
    if invnet
        # Ggpu isa Conv1x1gen && (Ggpu.init_weight! = true)
        # Gcpu isa Conv1x1gen && (Gcpu.init_weight! = true)
        Ycpu = randn(Float32, out_shape)
        Ygpu = gpu(deepcopy(Ycpu))
        @test cpu(Ggpu.inverse(Ygpu)) ≈ Gcpu.inverse(Ycpu) rtol=rtol
    end

    # Backward
    ΔYcpu = randn(Float32, out_shape)
    Ycpu = randn(Float32, out_shape)
    ΔYgpu = deepcopy(ΔYcpu) |> gpu
    Ygpu = deepcopy(Ycpu) |> gpu
    if invnet
        Xcpu = randn(Float32, input_shape)
        Xgpu = deepcopy(Xcpu) |> gpu
    end
    invnet ? ((ΔXgpu, Xgpu) = Ggpu.backward(ΔYgpu, Ygpu)) : (ΔXgpu = Ggpu.backward(ΔYgpu, Xgpu))
    Δθgpu = deepcopy(get_grads(Ggpu)) |> cpu
    ΔXgpu = ΔXgpu |> cpu
    Xgpu = Xgpu |> cpu
    invnet ? ((ΔXcpu, Xcpu) = Gcpu.backward(ΔYcpu, Ycpu)) : (ΔXcpu = Gcpu.backward(ΔYcpu, Xcpu))
    Δθcpu = deepcopy(get_grads(Gcpu))
    @test ΔXgpu ≈ ΔXcpu rtol=rtol
    @test Δθgpu ≈ Δθcpu rtol=rtol
    @test Xgpu ≈ Xcpu rtol=rtol

end

# function convert_params!(eltype::DataType, N::InvertibleNetwork)
#     θ = get_params(N)
#     for i = eachindex(θ)
#         θ[i].data = convert.(eltype, θ[i].data)
#         ~isnothing(θ[i].grad) && (θ[i].grad = convert.(eltype, θ[i].grad))
#     end
#     return θ
# end