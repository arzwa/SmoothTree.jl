abstract type SiteUpdateAlgorithm end

# EPABC in general
# ----------------
@with_kw mutable struct EPABC{A,Y,M} 
    data::Y  # data
    model::M  # global approximation
    sites::Vector{M}  # site approximations
    siteC::Vector{Float64}  # site normalization constants
    siteupdate::A  # site update algorithm
    λ::Float64 = 0.1  # for damped updates
    prunetol::Float64 = 1e-9  # prune low probability splits
end

"""
    ep!(alg, n; kwargs...)

Do `n` serial EP passes.
"""
ep!(alg, n=1; kwargs...) = mapreduce(i->ep_serial!(alg; kwargs...), vcat, 1:n)

# compute the model evidence
evidence(alg) = sum(alg.siteC) + logpartition(alg.model) - logpartition(getprior(alg))

# get the cavity wrt index i
getcavity(alg, i) = isassigned(alg.sites, i) ? alg.model - alg.sites[i] : alg.model

# get the prior
getprior(alg) = alg.sites[end]

# prune the model and site approximations
function prune!(alg)
    @unpack prunetol, sites = alg
    prunetol == 0. && return
    map(i->isassigned(sites, i) && prune!(sites[i], prunetol), 1:length(sites))
    alg.model = reduce(+, sites)
end

# update the model (main EP update)
function update!(alg, full, cavity, i, lZ)
    @unpack λ, prunetol = alg
    siteup = λ * (full - alg.model)
    prunetol != 0. && prune!(siteup, prunetol)  # should we?
    alg.sites[i] = isassigned(alg.sites, i) ? alg.sites[i] + siteup : siteup
    alg.model = alg.model + siteup
    if isfinite(lZ)
        alg.siteC[i] = lZ - logpartition(alg.model) + logpartition(cavity)
    end
    # I am not 100% sure about the normalizing constant -- how does this play
    # with λ? should we use the updated model or not?
end

# Kong's ESS estimator
ess(w) = 1/sum(w .^ 2) 

# obtain a probability vector in a numerically safish way
function lognormalize(ls)
   ps = exp.(ls .- maximum(ls))
   return ps ./ sum(ps)
end


# Importance Sampler
# ------------------
@with_kw mutable struct ImportanceSampler{P} <: SiteUpdateAlgorithm
    N::Int64  # number of particles
    sims::Vector{P}  # particles
    target::Float64 = 200. # target ESS
    miness::Float64 = 1.  # minimum ESS for succesful update
    α::Float64 = 1e-16  # Dirichlet-MBM prior in tilted approximation
end

# in the general case we cannot recycle gene trees...
struct Particle{T}
    S::Branches{T}  # species tree
    G::Splits{T}  # gene tree
    w::Float64  # sampler density
end

function EPABCIS(data, prior::MSCModel{T,V}, N; 
                 λ=0.1, prunetol=1e-9, kwargs...) where {T,V}
    sites = Vector{MSCModel{T,V}}(undef, length(data)+1)
    siteC = fill(-Inf, length(data))
    sites[end] = prior  # last entry is the prior
    sims = Vector{Particle{T}}(undef, N)
    siteupdate = ImportanceSampler(N=N, sims=sims; kwargs...)
    alg = EPABC(data=data, model=prior, sites=sites, siteC=siteC,
                siteupdate=siteupdate, λ=λ, prunetol=prunetol)
end

"""
    ep_serial!(alg; rnd=true)

Serial EP pass. When `rnd=true` the pass goes over the data in a random order. 
"""
function ep_serial!(alg::EPABC{<:ImportanceSampler}; rnd=true) 
    rnge = rnd ? shuffle(1:length(alg.data)) : 1:length(alg.data)
    iter = ProgressBar(rnge)
    ev = n = -Inf
    regen = true
    trace = map(iter) do i
        desc = string(@sprintf("%4d %8.1f %9.3f %2d", i, n, ev, regen))
        set_description(iter, desc)
        full, cavity, n, Z, regen = ep_iteration!(alg, i, regen)
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
function ep_iteration!(alg::EPABC{<:ImportanceSampler}, i, regen=true)
    @unpack siteupdate, data = alg
    @unpack sims, miness, N, α, target = siteupdate
    X = data[i]
    cavity = getcavity(alg, i)
    if regen
        simulate!(sims, cavity, X.init)
        w = zeros(N)
    else
        # in the general case we cannot recycle gene trees...
        simulate_genetrees!(sims, cavity, X.init)
        w = logweights(sims, cavity)
    end
    data_likelihood!(w, X.data, sims)
    W = lognormalize(w)
    n = ess(W)
    Z = logsumexp(w) - log(length(w))
    if n < miness
        full = alg.model
    else 
        branches = getfield.(sims, :S)
        full = matchmoments(branches, W, cavity, α)
    end
    regen = n < target
    return full, cavity, n, Z, regen
end

function data_likelihood!(w, data, sims)
    Threads.@threads for j=1:length(w)
        w[j] += logpdf(data, sims[j].G)
    end
end

function simulate!(particles, model, init)
    Threads.@threads for i=1:length(particles)
        particles[i] = simfun(model, init)
    end
end

function simfun(model, init)
    S = randbranches(model)
    G = randsplits(S, init)
    p = Particle(S, G, logpdf(model, S))
end

# resimulate gene trees for the same particles
function simulate_genetrees!(particles, model, init)
    Threads.@threads for i=1:length(particles)
        p = particles[i]
        G = randsplits(p.S, init)
        particles[i] = Particle(p.S, G, p.w)
    end
end

# XXX threaded?
function logweights(particles, model)
    map(p->logpdf(model, p.S) - p.w, particles)
end



