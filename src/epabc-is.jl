# issues:
# - the adaptive strategy could probably be optimized further...
# - the marginal likelihood is -Inf whenever there are data points with no
#   accepted simulations

# Kong estimator
ess(w) = 1/sum(w .^ 2) 

struct Particle{T,N,X}
    S::N  # species tree
    G::X  # gene tree
    w::T  # sampler density
end


"""
    EPABC(data, prior; kwargs...)

Expectation-propagation approximate Bayesian computation (EP-ABC, or
likelihood free EP) algorithm struct. See `ep!` and `pep!`.

## References
- Barthelmé, Simon, and Nicolas Chopin. 
  "Expectation propagation for likelihood-free inference." 
  Journal of the American Statistical Association 
  109.505 (2014): 315-333.
- Barthelmé, Simon, Nicolas Chopin, and Vincent Cottet. 
  "Divide and conquer in ABC: Expectation-propagation algorithms 
  for likelihood-free inference." 
  Handbook of Approximate Bayesian Computation. 
  Chapman and Hall/CRC, 2018. 415-434.
"""
@with_kw mutable struct EPABCIS{X,M,T,V,W} <: AbstractEPABC
    data ::X
    model::M
    sites::Vector{M}
    siteC::Vector{Float64}
    λ::Float64 = 0.1    # for damped update...
    α::Float64 = 1e-16  # Dirichlet-MBM parameter for 'moment matching'
    miness::Int = 10    # minimum ESS
    target::Int = 500   # target ESS
    maxsim::Int = 1e5 
    prunetol::Float64 = 1e-9
    sims ::Vector{Particle{T,V,W}}
end

function EPABCIS(data, prior::T; maxsim=1e5, kwargs...) where T
    sites = Vector{T}(undef, length(data)+1)
    siteC = fill(-Inf, length(data))
    sites[end] = prior  # last entry is the prior
    sims = simulate(prior, maxsim)
    alg = EPABCIS(data=data, model=prior, sites=sites, siteC=siteC, 
                  maxsim=maxsim, sims=sims; kwargs...)
    return alg
end

"""
    ep_serial!(alg; rnd=true)

Serial EP pass. When `rnd=true` the pass goes over the data in a
random order. 
"""
function ep_serial!(alg::EPABCIS; rnd=true) 
    rnge = rnd ? shuffle(1:length(alg.data)) : 1:length(alg.data)
    iter = ProgressBar(rnge)
    ev = n = ess_ws = -Inf
    trace = map(iter) do i
        desc = string(@sprintf("%8.1f; %8.1f; %9.3f", ess_ws, n, ev))
        set_description(iter, desc)
        full, cavity, n, Z, sims, ess_ws = ep_iteration(alg, i)
        alg.sims = sims
        update!(alg, full, cavity, i, Z) 
        ev = evidence(alg)
        alg.model, ev, n
    end 
    prune!(alg)
    return trace
end

"""
    ep_iteration(alg, i)

Do a single EP site update with recycling of simulations.
"""
function ep_iteration(alg::EPABCIS, i)
    @unpack model, sites, λ, α, miness, target, maxsim, sims = alg
    X = alg.data[i]
    cavity = getcavity(i, model, sites)
    ws = logweights(sims, cavity)
    Ws = exp.(ws)
    Ws /= sum(Ws)  # normalized weights
    ess_ws = ess(Ws)
    if ess_ws < alg.target  # fresh simulations required
        sims = simulate(cavity, alg.maxsim)
        ws = zeros(alg.maxsim)
    end
    ws .+= logpdf.(Ref(X), getfield.(sims, :G))
    full, n, Z = _momentmatching(alg, cavity, sims, ws)
    return full, cavity, n, Z, sims, ess_ws
end

function simulate(model, n)
    smpler = MSCSampler(model)
    init = idinit(randtree(smpler))
    tmap(_->simfun(model, smpler, init), 1:n)
end

function simfun(model, smpler, init)
    S = randtree(smpler)
    l = logpdf(model, S)
    G = randsplits(MSC(S, init))
    return Particle(S, G, l)
end

function logweights(particles, model)
    tmap(p->logpdf(model, p.S) - p.w, particles)
end

function _momentmatching(alg::EPABCIS, cavity, sims, lws)
    # importance weights
    w = exp.(lws)  # input are log weights
    W = w / sum(w)
    Z = mean(w)
    n = ess(W)
    n < alg.miness && return alg.model, n, Z
    trees = getfield.(sims, :S)
    full = matchmoments(trees, W, cavity, alg.α)
    return full, n, Z
end

