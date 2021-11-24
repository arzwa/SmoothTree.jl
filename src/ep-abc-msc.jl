# Explore the possibility of EP-ABC based inference under the MSC
# using the CCD as data: would have the advantage of
# - Bayesian inference
# - Takes into account gene tree uncertainty
# - Takes advantage of the fast simulation but intractable likelihood
#   situation

# not generic, but specifically for MSC-CCD application
mutable struct GaussianEPABC
    data     # the data
    S        # species tree
    prior_μ  # MVN prior mean
    prior_Σ  # MVN prior covariance
    tol      # ABC tolerance
    M        # number of simulations per iteration
    N        # number of trees simulated in each simulation
    μ        # current approximation mean
    Σ        # current approximation covariance
    Z        # current marginal likelihood
end

const DefaultNode = Node{Int64,NewickData{Float64,String}}

function initdict(model::MSC, y)
    spleaves = Dict(name(n)=>id(n) for n in model.l)
    init = Dict(v => DefaultNode[] for v in values(spleaves))
    for (i,gene) in enumerate(y.leaves)
        species = spleaves[_spname(gene)]
        push!(init[species], Node(i, n=gene))
    end
    return init
end
    
function ep_iteration(alg::GaussianEPABC, i)
    @unpack S, M, N, μ, Σ, tol, data = alg
    θ = rand(MvNormal(μ, Σ), M)
    y = data[i]
    init  = initdict(S.S, y)
    dists = map(1:M) do m
        setdistance!(S.S, exp.(θ[:,m]))
        #trees = proportionmap(trees)
        #kldiv(trees, y) 
        tree = randtree(S, copy(init))
        abs(logpdf(y, tree))
    end
    # compute empirical moments
    Z_ = 0.
    μ_ = zero(μ)
    Σ_ = zero(Σ)
    for (i, d) in enumerate(dists)
        d > tol && continue
        Z_ += 1
        μ_ += θ[:,i]
        Σ_ += θ[:,i]*θ[:,i]'
    end
    alg.Z = Z_/M
    alg.μ = μ_/Z_
    alg.Σ = Σ_/Z_ - alg.μ*alg.μ'
    return dists, θ
end

function kldiv(trees, ccd, ϵ=-Inf)
    kl = 0.
    for (tree, f) in trees
        lp = SmoothTree.logpdf(ccd, tree)
        lp = isfinite(lp) ? lp : ϵ
        kl += f * (log(f) - lp)
    end
    return kl
end


# UNIVARIATE CASE =================================================
# We approximate the posterior of the coalescent branch length which
# is shared by all branches by a Normal distribution.
mutable struct _UvNormalEPABC{T,V,U}
    model::U    # model (MSC)
    X::V        # data
    μ::T        # mean
    V::T        # variance
    Z::T        # marginal likelihood
    M::Int      # num simulations
    ϵ::Float64  # ABC tolerance
end

function ep_iteration!(alg::_UvNormalEPABC, i)
    @unpack X, μ, V, M, ϵ, model = alg
    θ = rand(Normal(μ, √V), M)
    x = X[i]
    init = initdict(model, x)
    sims = map(1:M) do m
        setdistance!(model.S, exp(θ[m]))
        tree = randtree(model, copy(init))
        exp(logpdf(x, tree))
        #trees = randtree(model, copy(init), 100)
        #trees = proportionmap(trees)
        #kldiv(trees, x) 
    end
    acc = 0
    μ_ = 0.
    V_ = 0.
    for (m,d) in enumerate(sims)
        #d > ϵ && continue
        rand() > d/ϵ && continue
        acc += 1
        μ_ += θ[m]
        V_ += θ[m]^2
    end
    if acc/M > 1e-3
        alg.Z = acc/M
        alg.μ = μ_/acc
        alg.V = V_/acc - alg.μ^2
        @info "Accepted: $(acc/M), μ=$(alg.μ), V=$(alg.V)"
    end
    return sims, θ
end



