function gradient_test_input(G, loss::Function, X::AbstractArray{T}; step::T=1f-4, rtol::T=1f-3) where T

    # Computing gradients
    Y = G.forward(X)
    l, ΔY = loss(Y)
    ΔX = G.backward(ΔY, X)

    # Perturbations
    dX = randn(T, size(X)); dX .*= norm(X)/norm(dX)

    # Test (wrt input)
    lp1, _ = loss(G.forward(X+T(0.5)*step*dX))
    lm1, _ = loss(G.forward(X-T(0.5)*step*dX))
    @test (lp1-lm1)/step ≈ dot(ΔX, dX) rtol=rtol

end

function gradient_test_pars(G, loss::Function, X::AbstractArray{T}; step::T=1f-4, rtol::T=1f-3) where T

    # Collecting parameters
    θ = deepcopy(get_params(G))

    # Computing gradients
    Y = G.forward(X)
    l, ΔY = loss(Y)
    Δθ = get_grads(G)

    # Perturbations
    dθ = Array{Parameter,1}(undef, length(θ))
    for i = 1:length(θ)
        dθ[i] = Parameter(randn(T, size(θ[i].data)))
        norm(θ[i].data) != T(0) && (dθ[i].data .*= norm(θ[i].data)/norm(dθ[i].data))
    end

    # Test (wrt pars)
    set_params!(G, θ+T(0.5)*step*dθ)
    lp1, _ = loss(G.forward(X))
    set_params!(G, θ-T(0.5)*step*dθ)
    lm1, _ = loss(G.forward(X))
    @test (lp1-lm1)/step ≈ dot(Δθ, dθ) rtol=rtol

end