function gradient_test_input(G, loss::Function, X::AbstractArray{T}; step::T=1f-4, rtol::T=1f-3, invnet::Bool=true) where T

    # Computing gradients
    G isa Conv1x1gen && (G.init_weight! = true)
    G.logdet ? ((Y, _) = G.forward(X)) : (Y = G.forward(X))
    _, ΔY = loss(Y)
    invnet ? ((ΔX, _) = G.backward(ΔY, Y)) : (ΔX = G.backward(ΔY, X))

    # Perturbations
    dX = randn(T, size(X)); dX .*= norm(X)/norm(dX)

    # Test (wrt input)
    G isa Conv1x1gen && (G.init_weight! = true)
    G.logdet ? ((Yp1, logdet_p1) = G.forward(X+T(0.5)*step*dX)) : (Yp1 = G.forward(X+T(0.5)*step*dX))
    lp1, _ = loss(Yp1)
    G.logdet && (lp1 -= logdet_p1)
    G isa Conv1x1gen && (G.init_weight! = true)
    G.logdet ? ((Ym1, logdet_m1) = G.forward(X-T(0.5)*step*dX)) : (Ym1 = G.forward(X-T(0.5)*step*dX))
    lm1, _ = loss(Ym1)
    G.logdet && (lm1 -= logdet_m1)
    @test (lp1-lm1)/step ≈ dot(ΔX, dX) rtol=rtol

end

function gradient_test_pars(G, loss::Function, X::AbstractArray{T}; step::T=1f-4, rtol::T=1f-3, invnet::Bool=true) where T

    # Collecting parameters
    θ = deepcopy(get_params(G))

    # Computing gradients
    G isa Conv1x1gen && (G.init_weight! = true)
    G.logdet ? ((Y, _) = G.forward(X)) : (Y = G.forward(X))
    _, ΔY = loss(Y)
    invnet ? ((ΔX, _) = G.backward(ΔY, Y)) : (ΔX = G.backward(ΔY, X))
    Δθ = deepcopy(get_grads(G))

    # Perturbations
    dθ = Array{Parameter,1}(undef, length(θ))
    for i = 1:length(θ)
        dθ[i] = Parameter(randn(T, size(θ[i].data)))
        norm(θ[i].data) != T(0) && (dθ[i].data .*= norm(θ[i].data)/norm(dθ[i].data))
    end

    # Test (wrt pars)
    set_params!(G, θ+T(0.5)*step*dθ)
    G isa Conv1x1gen && (G.init_weight! = true)
    G.logdet ? ((Yp1, logdet_p1) = G.forward(X)) : (Yp1 = G.forward(X))
    lp1, _ = loss(Yp1)
    G.logdet && (lp1 -= logdet_p1)
    set_params!(G, θ-T(0.5)*step*dθ)
    G isa Conv1x1gen && (G.init_weight! = true)
    G.logdet ? ((Ym1, logdet_m1) = G.forward(X)) : (Ym1 = G.forward(X))
    lm1, _ = loss(Ym1)
    G.logdet && (lm1 -= logdet_m1)
    @test (lp1-lm1)/step ≈ dot(Δθ, dθ) rtol=rtol

end

function cpu_vs_gpu_test(G, input_shape; rtol::Float32=1f-5)

    Ggpu = deepcopy(gpu(G))
    Gcpu = deepcopy(cpu(G))

    # Forward
    Xcpu = randn(Float32, input_shape)
    Xgpu = copy(Xcpu) |> gpu
    G.logdet ? ((Ygpu, logdet_gpu) = cpu(Ggpu.forward(Xgpu))) : (Ygpu = cpu(Ggpu.forward(Xgpu)))
    G.logdet ? ((Ycpu, logdet_cpu) = Gcpu.forward(Xcpu)) : (Ycpu = (Gcpu.forward(Xcpu)))
    @test Ygpu ≈ Ycpu rtol=rtol
    G.logdet && (@test logdet_gpu ≈ logdet_cpu rtol=rtol)

    # Inverse
    Ycpu = randn(Float32, input_shape)
    Ygpu = copy(Ycpu) |> gpu
    @test cpu(Ggpu.inverse(Ygpu)) ≈ Gcpu.inverse(Ycpu) rtol=rtol

    # Backward
    ΔYcpu = randn(Float32, input_shape)
    Ycpu = randn(Float32, input_shape)
    Ygpu = Ycpu |> gpu
    ΔYgpu = ΔYcpu |> gpu
    ΔXgpu, Xgpu = Ggpu.backward(ΔYgpu, Ygpu)
    Δθgpu = deepcopy(get_grads(Ggpu)) |> cpu
    ΔXgpu = ΔXgpu |> cpu
    Xgpu = Xgpu |> cpu
    ΔXcpu, Xcpu = Gcpu.backward(ΔYcpu, Ycpu)
    Δθcpu = deepcopy(get_grads(Gcpu))
    @test ΔXgpu ≈ ΔXcpu rtol=rtol
    @test Δθgpu ≈ Δθcpu rtol=rtol
    @test Xgpu ≈ Xcpu rtol=rtol

end
