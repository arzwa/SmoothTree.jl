# issues:
# - the adaptive strategy could probably be optimized further...
# - the marginal likelihood is -Inf whenever there are data points with no
#   accepted simulations

# Kong estimator
ess(w) = 1/sum(w .^ 2) 

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
@with_kw mutable struct EPABCIS{X,M} <: AbstractEPABC
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
end

function EPABCIS(data, prior::T; kwargs...) where T
    sites = Vector{T}(undef, length(data)+1)
    siteC = fill(-Inf, length(data))
    sites[end] = prior  # last entry is the prior
    alg = EPABCIS(data=data, model=prior, sites=sites, siteC=siteC; kwargs...)
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
    ev = n = -Inf
    trace = map(iter) do i
        desc = string(@sprintf("ESS = %8.1f; Z = %9.3f", n, ev))
        set_description(iter, desc)
        full, cavity, n, Z = ep_iteration(alg, i)
        update!(alg, full, cavity, i, Z) 
        ev = evidence(alg)
        alg.model, ev, n
    end 
    prune!(alg)
    return trace
end
 
"""
    ep_iteration(alg, i)

Do a single EP site update.
"""
function ep_iteration(alg::EPABCIS, i)
    @unpack model, sites, λ, α, miness, target, maxsim = alg
    X = alg.data[i]
    cavity = getcavity(i, model, sites)
    smpler = MSCSampler(cavity)
    sptree = randtree(smpler)
    init = idinit(sptree)
    sims = _inner_threaded(X, smpler, init, alg)
    full, n, Z = _momentmatching(alg, cavity, sims)
    return full, cavity, n, Z
end

"""
    _inner_threaded(...)

See `_inner_serial`, but in a multi-threaded implementation using a
user-defined batch size for simulation batches executed in parallel.
"""
function _inner_threaded(X, smpler, init, alg)
    tmap(_->simfun(smpler, init, X), 1:alg.maxsim)
end

function simfun(smpler, init, X)
    S = randtree(smpler)
    G = randsplits(MSC(S, init))
    l = logpdf(X, G)
    (l, S)
end

function _momentmatching(alg::EPABCIS, cavity, sims)
    # importance weights
    w = exp.(first.(sims))
    W = w / sum(w)
    Z = mean(w)
    n = ess(W)
    n < alg.miness && return alg.model, n, Z
    trees = last.(sims)
    full = matchmoments(trees, W, cavity, alg.α)
    return full, n, Z
end
