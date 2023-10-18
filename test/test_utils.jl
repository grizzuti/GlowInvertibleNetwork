function gradient_test_input(G, X::AbstractArray{T}; loss=x->(norm(x)^2/2, x), step::T=1f-4, rtol::T=1f-3, invnet::Bool=true) where T

    # Computing gradients
    if G isa InvertibleNetworks.ReversedNetwork
        hasfield(typeof(G.I), :logdet) ? (logdet = G.logdet) : (logdet = false)
    else
        hasfield(typeof(G), :logdet) ? (logdet = G.logdet) : (logdet = false)
    end
    logdet ? ((Y, _) = G.forward(X)) : (Y = G.forward(X))
    _, ΔY = loss(Y)
    invnet ? ((ΔX, _) = G.backward(ΔY, Y)) : (ΔX = G.backward(ΔY, X))

    # Perturbations
    dX = typeof(X)(randn(T, size(X))); dX .*= norm(X)/norm(dX)

    # Test (wrt input)
    logdet ? ((Yp1, logdet_p1) = G.forward(X+step*dX/2)) : (Yp1 = G.forward(X+step*dX/2))
    lp1, _ = loss(Yp1)
    logdet && (lp1 -= logdet_p1)
    logdet ? ((Ym1, logdet_m1) = G.forward(X-step*dX/2)) : (Ym1 = G.forward(X-step*dX/2))
    lm1, _ = loss(Ym1)
    logdet && (lm1 -= logdet_m1)
    @test isapprox((lp1-lm1)/step, dot(ΔX, dX); rtol=rtol)

end

function gradient_test_pars(G, X::AbstractArray{T}; loss=x->(norm(x)^2/2, x), step::T=1e-4, rtol::T=1e-3, invnet::Bool=true, dθ::Union{Nothing,Array{Parameter,1}}=nothing) where T

    # Perturbations
    G = deepcopy(G); θ = deepcopy(get_params(G))
    if isnothing(dθ)
        dθ = Vector{Parameter}(undef, length(θ))
        for i = eachindex(θ)
            if ~isnothing(θ[i].data)
                dθ[i] = Parameter(typeof(θ[i].data)(randn(T, size(θ[i].data))))
                norm(θ[i].data) != T(0) && (dθ[i].data .*= norm(θ[i].data)/norm(dθ[i].data))
            else
                dθ[i] = Parameter(nothing)
            end
        end
    end

    # Computing gradients
    if G isa InvertibleNetworks.ReversedNetwork
        hasfield(typeof(G.I), :logdet) ? (logdet = G.logdet) : (logdet = false)
    else
        hasfield(typeof(G), :logdet) ? (logdet = G.logdet) : (logdet = false)
    end
    logdet ? ((Y, _) = G.forward(X)) : (Y = G.forward(X))
    _, ΔY = loss(Y)
    invnet ? ((ΔX, _) = G.backward(ΔY, Y)) : (ΔX = G.backward(ΔY, X))
    Δθ = deepcopy(get_grads(G))

    # Test (wrt pars)
    set_params!(G, θ+step*dθ/2)
    logdet ? ((Yp1, logdet_p1) = G.forward(X)) : (Yp1 = G.forward(X))
    lp1, _ = loss(Yp1)
    logdet && (lp1 -= logdet_p1)

    set_params!(G, θ-step*dθ/2)
    logdet ? ((Ym1, logdet_m1) = G.forward(X)) : (Ym1 = G.forward(X))
    lm1, _ = loss(Ym1)
    logdet && (lm1 -= logdet_m1)
    @test isapprox((lp1-lm1)/step, dot(Δθ, dθ); rtol=rtol)

end