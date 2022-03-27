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
end

# compute the model evidence
function evidence(alg) 
    sum(alg.siteC) + logpartition(alg.model) - logpartition(getprior(alg))
end

function evidence_upperbound(alg)
    sum(filter(isfinite, alg.siteC)) + logpartition(alg.model) - logpartition(getprior(alg))
end

# get the prior
getprior(alg) = alg.sites[end]


# Construct cavity distribution
# -----------------------------
# Non-mutating
getcavity(alg, i) = isassigned(alg.sites, i) ? alg.model - alg.sites[i] : deepcopy(alg.model)
uncavity(alg, i)  = isassigned(alg.sites, i) ? alg.model + alg.sites[i] : deepcopy(alg.model)

# With mutation
getcavity!(alg, i) = isassigned(alg.sites, i) ? sub!(alg.model, alg.sites[i]) : alg.model
uncavity!(alg, i)  = isassigned(alg.sites, i) ? add!(alg.model, alg.sites[i]) : alg.model


# Site and model update
# ---------------------
function update!(alg, epresult)
    @unpack i, new_approx, Zhat, Φ_cavity = epresult
    update_model!(alg, i, new_approx)
    update_marginal!(alg, i, Zhat, Φ_cavity)
end

function update_model!(alg::EPABC, i, new_approx)
    # make sure that the new approximation is a distinct object.
    @assert !(alg.model === new_approx)
    # here we mutate `new_approx` to get the site update, which is the only
    # thing we will need in the remainder
    siteup = mul!(sub!(new_approx, alg.model), alg.λ)   # λ(q∗ - q)
    alg.prunetol != 0. && prune!(siteup, alg.prunetol)  # should we?
    add!(alg.model, siteup)  # mutates the model, to get the new approximation
    update_site!(alg.sites, i, siteup)
end

function update_site!(sites, i, siteup)
    sites[i] = isassigned(sites, i) ? add!(sites[i], siteup) : siteup
end

function update_marginal!(alg::EPABC, i, Zhat, Φ_cavity)
    Φ_model = logpartition(alg.model)
    alg.siteC[i] = Zhat - Φ_model + Φ_cavity
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

# A struct for a weighted particle
mutable struct Particle{X}
    S::X        # species tree
    w::Float64  # sampler density
end

# undef initialized particle
Particle(n::Int, ::Type{T}) where T = Particle(Branches(undef, T, n), 0.)

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
    result = ep_iteration!(alg, iter[1], true)
    result = (result..., ev=-Inf)
    trace = typeof(result)[]
    for (j, i) in enumerate(iter)
        report = [result.neff, result.ev, result.nacc, result.refresh]
        desc = string(@sprintf("%4d %8.1f %9.3f %6d %2d", i, report...))
        set_description(iter, desc)
        result = ep_iteration!(alg, i, result.refresh)
        !result.fail && update!(alg, result)
        result = (result..., ev=evidence(alg))
        push!(trace, result)
    end 
    return trace
end

"""
    ep_iteration(alg, i)

Do a single EP site update with recycling of simulations.
"""
function ep_iteration!(alg::EPABC{<:ImportanceSampler}, i, refresh=true)
    @unpack siteupdate, data = alg
    @unpack sims, miness, N, α, target, c = siteupdate
    X = data[i]
    getcavity!(alg, i)
    Φ_cavity = logpartition(alg.model)
    if refresh
        simulate!(sims, alg.model)
        w = zeros(N)
    else
        w = logweights(sims, alg.model)
    end
    # note that in the general case we cannot recycle gene trees
    data_likelihood!(w, X, sims)
    W, neff, Zhat, accept = process_weights(w, c)
    if neff < miness || !isfinite(Zhat) 
        fail = true
        new_approx = alg.model
    else 
        branches = alltrees(sims)[accept]
        new_approx = matchmoments(branches, W, alg.model, α)
        fail = false
    end
    uncavity!(alg, i)
    regen = neff < target || !isfinite(Zhat) || !isfinite(neff)
    epresult = (i=i, 
                new_approx=new_approx, 
                neff=neff, 
                Zhat=Zhat,
                Φ_cavity=Φ_cavity,
                refresh=regen, 
                nacc=sum(accept), 
                fail=fail)
    return epresult
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
        G = randsplits(sims[j].S, getinit(data))  # coalescent sim
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


# Inference for fixed topology
# ============================
# For a fixed topology, the following should do. We simply use the BranchModel
# instead of the MSCModel, and let randbranches! only draw branch lengths.

# This needs to be improved, to allow for Multivariate Gaussian approximating
# families...
const BranchOnly = Union{<:BranchModel,<:NaturalMvNormal}

function EPABCIS(data, b::Branches{T,V}, prior::M, N; 
                 λ=0.1, prunetol=0., kwargs...) where {T,V,M<:BranchOnly}
    sites = Vector{M}(undef, length(data)+1)
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


# Sequential importance sampling
# ==============================

@with_kw mutable struct SIS{P} <: SiteUpdateAlgorithm
    N::Int64  # number of particles
    M::Int64  # number iterations
    sims::Vector{P}  # particles
    target::Float64 = 100. # target ESS
    miness::Float64 = 2.  # minimum ESS for succesful update
    α::Float64 = 1e-3  # Dirichlet-MBM prior in tilted approximation
    c::Float64 = 0.0   # rejection control quantile
end

function EPABCSIS(data, prior::MSCModel{T,V}, N, M; 
                  λ=0.1, prunetol=0., kwargs...) where {T,V}
    sites = Vector{MSCModel{T,V}}(undef, length(data)+1)
    siteC = fill(-Inf, length(data))
    sites[end] = prior  # last entry is the prior
    ntaxa = Int(log2(prior.S.root + 1))
    nbranch = 2ntaxa - 2
    sims = map(_->Particle(nbranch, T), 1:N)
    siteupdate = SIS(N=N, M=M, sims=sims; kwargs...)
    alg = EPABC(data=data, model=deepcopy(prior), sites=sites, siteC=siteC,
                siteupdate=siteupdate, λ=λ, prunetol=prunetol)
end

function EPABCSIS(data, b::Branches, prior::X, N, M; 
                 λ=0.1, prunetol=0., kwargs...) where {T,V,X<:BranchOnly}
    sites = Vector{X}(undef, length(data)+1)
    siteC = fill(-Inf, length(data))
    sites[end] = prior  # last entry is the prior
    sims = map(_->Particle(copy(b), -Inf), 1:N)
    siteupdate = SIS(N=N, M=M, sims=sims; kwargs...)
    alg = EPABC(data=data, model=deepcopy(prior), sites=sites, siteC=siteC,
                siteupdate=siteupdate, λ=λ, prunetol=prunetol)
end

function ep_serial!(alg::EPABC{<:SIS}; rnd=true, traceit=10000) 
    rnge = rnd ? shuffle(1:length(alg.data)) : 1:length(alg.data)
    iter = ProgressBar(rnge)
    result = ep_iteration!(alg, iter[1])
    result = (result..., ev=-Inf)
    trace = typeof(result)[]
    for (j, i) in enumerate(iter)
        report = [result.neff, result.ev, result.nacc, result.fail, result.m]
        desc = string(@sprintf("%4d %8.1f %9.3f %6d %2d %2d", i, report...))
        set_description(iter, desc)
        result = ep_iteration!(alg, i)
        !result.fail && update!(alg, result)
        result = (result..., ev=evidence(alg))
        push!(trace, result)
    end 
    return trace
end

function ep_iteration!(alg::EPABC{<:SIS}, i)
    @unpack siteupdate, data = alg
    @unpack sims, miness, M, N, α, target, c = siteupdate
    X = data[i]
    getcavity!(alg, i)
    Φ_cavity = logpartition(alg.model)
    g, rest = sis_inner!(sims, alg.model, alg.model, X, α, c, miness) 
    m = 1
    while rest.neff < target && m < M
        g, rest = sis_inner!(sims, g, alg.model, X, α, c, miness) 
        m += 1
    end
    uncavity!(alg, i)
    fail = rest.neff < miness || !isfinite(rest.Zhat)
    result = (rest..., m=m, i=i, new_approx=g, Φ_cavity=Φ_cavity, fail=fail)
    return result
end

function sis_inner!(sims, g, prior, X, α, c, miness)
    simulate!(sims, g)
    #w = map(p->logpdf(prior, p.S) - p.w, sims)
    w = logweights(sims, prior)
    data_likelihood!(w, X, sims)
    W, neff, Zhat, accept = process_weights(w, c)
    branches = alltrees(sims)[accept]
    if neff < miness
        g′ = prior
    else
        g′ = matchmoments(branches, W, g, α)
    end
    g′, (neff=neff, Zhat=Zhat, nacc=sum(accept))
end

