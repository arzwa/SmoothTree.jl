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
    prunetol::Float64 = 0.  # prune low probability splits
end

"""
    ep!(alg, n; kwargs...)

Do `n` serial EP passes.
"""
function ep!(alg, n=1; kwargs...) 
    trace = mapreduce(i->ep_serial!(alg; kwargs...), vcat, 1:n)
    mtrace = vcat(first.(trace)...)
    ztrace = vcat(last.(trace)...)
    return mtrace, ztrace
end

# compute the model evidence
function evidence(alg) 
    sum(alg.siteC) + logpartition(alg.model) - logpartition(getprior(alg))
end

# get the cavity wrt index i
getcavity(alg, i) = isassigned(alg.sites, i) ? 
    alg.model - alg.sites[i] : deepcopy(alg.model)
uncavity(alg, i)  = isassigned(alg.sites, i) ? 
    alg.model + alg.sites[i] : deepcopy(alg.model)

# take out site i
getcavity!(alg, i) = isassigned(alg.sites, i) ? 
    sub!(alg.model, alg.sites[i]) : alg.model
uncavity!(alg, i)  = isassigned(alg.sites, i) ? 
    add!(alg.model, alg.sites[i]) : alg.model

# get the prior
getprior(alg) = alg.sites[end]

# prune the model and site approximations
# useful ?
function prune!(alg)
    @unpack prunetol, sites = alg
    prunetol == 0. && return
    map(i->isassigned(sites, i) && prune!(sites[i], prunetol), 1:length(sites))
    # inefficient
    alg.model = reduce(+, sites)
end

# update the model (main EP update)
#function update!(alg, full, cavity, i, lZ)
#    @unpack λ, prunetol = alg
#    # note we can do all model operations in a mutating fashion. The new model
#    # `full` is independent in each generation.
#    siteup = λ*(full - alg.model)
#    prunetol != 0. && prune!(siteup, prunetol)  # should we?
#    if isassigned(alg.sites, i) 
#        alg.sites[i] = alg.sites[i] + siteup
#    else siteup
#        alg.sites[i] = siteup
#    end
#    alg.model = alg.model + siteup
#    if isfinite(lZ)
#        alg.siteC[i] = lZ - logpartition(alg.model) + logpartition(cavity)
#    end
#    # I am not 100% sure about the normalizing constant -- how does this play
#    # with λ? should we use the updated model or not?
#end

# XXX alg.model should be the cavity when calling this function!
#     that is quite bug-prone... better separate the marginal likelihood update
#     from the model update... so that we don't need to compute the cavity
#     partition function here...
function update!(alg::EPABC, full, i, lZ)
    @unpack λ, prunetol = alg
    @assert !(alg.model === full)
    Zcav = isfinite(lZ) ? logpartition(alg.model) : -Inf
    # here we mutate `full` to get `siteup`, which is the only thing we will
    # need in the remainder (we no longer need full)
    uncavity!(alg, i)
    siteup = mul!(sub!(full, alg.model), λ)   # λ(q∗ - q)
    prunetol != 0. && prune!(siteup, prunetol)  # should we?
    add!(alg.model, siteup)  # mutates the model, to get the new approximation
    if isassigned(alg.sites, i) 
        add!(alg.sites[i], siteup)   # here we mutate the site, no problem
    else siteup
        alg.sites[i] = siteup  # new site, siteup is an independent object 
        # since derived from `full`
    end
    if isfinite(lZ)
        alg.siteC[i] = lZ - logpartition(alg.model) + Zcav
    end
end


# Importance Sampler
# ------------------
@with_kw mutable struct ImportanceSampler{P} <: SiteUpdateAlgorithm
    N::Int64  # number of particles
    sims::Vector{P}  # particles
    target::Float64 = 100. # target ESS
    miness::Float64 = 2.  # minimum ESS for succesful update
    α::Float64 = 1e-3  # Dirichlet-MBM prior in tilted approximation
    c::Float64 = 0.0   # rejection control quantile
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
    alg = EPABC(data=data, model=deepcopy(prior), sites=sites, siteC=siteC,
                siteupdate=siteupdate, λ=λ, prunetol=prunetol)
end

"""
    ep_serial!(alg; rnd=true)

Serial EP pass. When `rnd=true` the pass goes over the data in a random order. 
"""
function ep_serial!(alg::EPABC{<:ImportanceSampler}; rnd=true, traceit=10000) 
    rnge = rnd ? shuffle(1:length(alg.data)) : 1:length(alg.data)
    iter = ProgressBar(rnge)
    ev = n = -Inf
    accept = 0
    regen = true
    mtrace = typeof(alg.model)[]
    Ztrace = Tuple{Float64,Float64}[]
    for (j, i) in enumerate(iter)
        desc = string(@sprintf("%4d %8.1f %9.3f %6d %2d", i, n, ev, accept, regen))
        set_description(iter, desc)
        full, n, Z, regen, accept, fail = ep_iteration!(alg, i, regen)
        !fail && update!(alg, full, i, Z) 
        ev = evidence(alg)
        j % traceit == 0 && push!(mtrace, deepcopy(alg.model))
        push!(Ztrace, (ev, n))
    end 
    return mtrace, Ztrace
end

"""
    ep_iteration(alg, i)

Do a single EP site update with recycling of simulations.
"""
function ep_iteration!(alg::EPABC{<:ImportanceSampler}, i, regen=true)
    @unpack siteupdate, data = alg
    @unpack sims, miness, N, α, target, c = siteupdate
    X = data[i]
    #cavity = getcavity(alg, i)
    getcavity!(alg, i)
    if regen
        simulate!(sims, alg.model)
        w = zeros(N)
    else
        w = logweights(sims, alg.model)
    end
    # note that in the general case we cannot recycle gene trees
    data_likelihood!(w, X, sims)
    W, n, Z, accept = process_weights(w, c)
    if n < miness || !isfinite(Z) 
        full = uncavity!(alg, i)
        fail = true
    else 
        branches = alltrees(sims)[accept]
        full = matchmoments(branches, W, alg.model, α)
        fail = false
    end
    # Note that `full` must be an independent object from the current model. At
    # least, this is assumed in `update!`.
    regen = n < target || !isfinite(Z) || !isfinite(n)
    return full, n, Z, regen, sum(accept), fail
end

alltrees(xs::Vector{Particle{T}}) where T = map(x->x.S, xs)

function logweights(particles, model)
    w = zeros(length(particles))
    Threads.@threads for i=1:length(particles)
        p = particles[i]
        w[i] = logpdf(model, p.S) - p.w
    end
    return w
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

_accfun(x) = x == 0. ? true : log(rand()) < x
function process_weights(w, c)  # with rejection control
    Z  = logsumexp(w) - log(length(w))  # marginal likelihood estimator
    wc = quantile(w, c)
    r  = min.(w .- wc, Ref(0.))
    #accept = log.(rand(length(w))) .< r  # generates too many rns...
    accept = _accfun.(r)
    sum(accept) <= 1 && return w, 0., -Inf, accept
    w .-= r  # modified weight
    W = lognormalize(w[accept])  # normalized importance weights
    n = ess(W)  # ESS estimator (Kong)
    return W, n, Z, accept
end

function simulate!(particles, model::MSCModel)
    # it is quite a bit faster to convert to MomMBM on beforehand
    m = tomoment(model.S)
    Threads.@threads for i=1:length(particles)
        simfun!(particles[i], model, m)
    end
end

function simfun!(p, model, m)
    randbranches!(p.S, m, model.ϕ)
    p.w = logpdf(model, p.S)
end

# currently not used (mutating instead)
# note this one actually also mutates the particles array, but not the
# particles themselves...
#function simulate(particles, model::MSCModel)
#    m = MomMBM(model.S)
#    Threads.@threads for i=1:length(particles)
#        particles[i] = simfun(model, m)
#    end
#end
#
#function simfun(model, m)
#    S = randbranches(m, model.q)
#    p = Particle(S, logpdf(model, S))
#end

# For a fixed topology, the following should do. We simply use the BranchModel
# instead of the MSCModel, and let randbranches! only draw 
function EPABCIS(data, b::Branches, prior::BranchModel{T,V}, N; 
                 λ=0.1, prunetol=1e-9, kwargs...) where {T,V}
    sites = Vector{BranchModel{T,V}}(undef, length(data)+1)
    siteC = fill(-Inf, length(data))
    sites[end] = prior  # last entry is the prior
    sims = map(_->Particle(copy(b), -Inf), 1:N)
    siteupdate = ImportanceSampler(N=N, sims=sims; kwargs...)
    alg = EPABC(data=data, model=prior, sites=sites, siteC=siteC,
                siteupdate=siteupdate, λ=λ, prunetol=prunetol)
end

function simulate!(particles, model::BranchModel)
    Threads.@threads for i=1:length(particles)
        simfun!(particles[i], model)
    end
end

function simfun!(p, model::BranchModel)
    randbranches!(p.S, model)
    p.w = logpdf(model, p.S)
end

function randbranches!(branches, q::BranchModel)
    for (i,x) in enumerate(branches)
        d = randbranch(q, x[1], x[2])
        branches[i] = (x[1], x[2], d)
    end
end

@with_kw mutable struct SIS{P} <: SiteUpdateAlgorithm
    N::Int64  # number of particles
    sims::Vector{P}  # particles
    target::Float64 = 100. # target ESS
    miness::Float64 = 2.  # minimum ESS for succesful update
    α::Float64 = 1e-3  # Dirichlet-MBM prior in tilted approximation
    c::Float64 = -Inf  # rejection control using quantile
end


