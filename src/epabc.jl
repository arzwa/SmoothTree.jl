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


# Importance Sampler
# ------------------
@with_kw mutable struct ImportanceSampler{P} <: SiteUpdateAlgorithm
    N::Int64  # number of particles
    sims::Vector{P}  # particles
    target::Float64 = 100. # target ESS
    miness::Float64 = 2.  # minimum ESS for succesful update
    α::Float64 = 1e-3  # Dirichlet-MBM prior in tilted approximation
end

# in the general case we cannot recycle gene trees... so don't store them
mutable struct Particle{T}
    S::Branches{T}  # species tree
    w::Float64  # sampler density
end

# undef initialized particle
Particle(n::Int, ::Type{T}) where T = Particle(Branches{T}(undef, n), 0.)

# Kong's ESS estimator
ess(w) = 1/sum(w .^ 2) 

# obtain a probability vector in a numerically safish way
function lognormalize(ls)
   ps = exp.(ls .- maximum(ls))
   return ps ./ sum(ps)
end

"""
    EPABCIS(data, prior, nparticles; kwargs...)

Likelihood-free expectation propagation with an importance sampling step to do
the site update.
"""
function EPABCIS(data, prior::MSCModel{T,V}, N; 
                 λ=0.1, prunetol=1e-9, kwargs...) where {T,V}
    sites = Vector{MSCModel{T,V}}(undef, length(data)+1)
    siteC = fill(-Inf, length(data))
    sites[end] = prior  # last entry is the prior
    #sims = Vector{Particle{T}}(undef, N)
    ntaxa = Int(log2(prior.S.root + 1))
    nbranch = 2ntaxa - 2
    sims = map(_->Particle(nbranch, T), 1:N)
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
        if isfinite(Z)
            update!(alg, full, cavity, i, Z) 
        else
            @warn "Z ($Z) not finite"
        end
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
        simulate!(sims, cavity)
        w = zeros(N)
    else
        w = logweights(sims, cavity)
    end
    # note that in the general case we cannot recycle gene trees
    data_likelihood!(w, X, sims)
    W, n, Z = process_weights(w)
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
        G = randsplits(sims[j].S, data.init)  # coalescent sim
        w[j] += logpdf(data.data, G)  # kernel
    end
end

function process_weights(w)
    W = lognormalize(w)  # normalized importance weights
    n = ess(W)  # ESS estimator (Kong)
    Z = logsumexp(w) - log(length(w))  # marginal likelihood estimator
    return W, n, Z
end

function simulate!(particles, model)
    # it is quite a bit faster to convert to MomMBM on beforehand
    m = MomMBM(model.S)
    Threads.@threads for i=1:length(particles)
        simfun!(particles[i], model, m)
    end
end

function simfun!(p, model, m)
    randbranches!(p.S, m, model.q)
    p.w = logpdf(model, p.S)
end

# currently not used (mutating instead)
# note this one actually also mutates the particles array, but not the
# particles themselves...
function simulate(particles, model)
    m = MomMBM(model.S)
    Threads.@threads for i=1:length(particles)
        particles[i] = simfun(model, m)
    end
end

function simfun(model, m)
    S = randbranches(m, model.q)
    p = Particle(S, logpdf(model, S))
end

function logweights(particles, model)
    w = zeros(length(particles))
    Threads.@threads for i=1:length(particles)
        p = particles[i]
        w[i] = logpdf(model, p.S) - p.w
    end
    return w
end

