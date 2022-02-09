# issues:
# - the adaptive strategy could probably be optimized further...
# - the marginal likelihood is -Inf whenever there are data points with no
#   accepted simulations

abstract type AbstractEPABC end

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
@with_kw mutable struct EPABC{X,M} <: AbstractEPABC
    data ::X
    model::M
    sites::Vector{M}
    siteC::Vector{Float64}
    λ::Float64 = 0.1    # for damped update...
    α::Float64 = 1e-16  # Dirichlet-MBM parameter for 'moment matching'
    minacc::Int = 10
    target::Int = 500
    maxsim::Int = 1e5 
    h::Float64 = 1.
    ν::Float64 = 0.1
    tuneh::Bool = true
    batch::Int = 1
    prunetol::Float64 = 1e-9
end

function tuneoff!(alg)
    alg.h = 1.
    alg.tuneh = false
end

function EPABC(data, prior::T; kwargs...) where T
    sites = Vector{T}(undef, length(data)+1)
    siteC = fill(-Inf, length(data))
    sites[end] = prior  # last entry is the prior
    alg = EPABC(data=data, model=prior, sites=sites, siteC=siteC; kwargs...)
    return alg
end

# get an initial estimate for `h`
function initialize_h!(alg)
    sims = randtree(alg.model, alg.maxsim)    
    init = idinit(sims[1])
    ys = map(m->randsplits(SmoothTree.MSC(m, init)), sims)
    ls = map(alg.data) do x
        exp(logsumexp(map(y->logpdf(x,y), ys))) / length(ys)
    end 
    alg.h = alg.target/(alg.maxsim * mean(ls))
end

# get the prior
prior(alg) = alg.sites[end]

# get the cavity distribution
getcavity(i, m, s) = isassigned(s, i) ? m - s[i] : m

"""
    ep!(alg, n; kwargs...)

Do `n` serial EP passes.
"""
ep!(alg, n=1; kwargs...) = mapreduce(i->ep_serial!(alg; kwargs...), vcat, 1:n)

"""
    pep!(alg, n)

Do `n` (embarrassingly) parallel EP passes.
"""
pep!(alg, n=1; kwargs...) = mapreduce(i->ep_parallel!(alg; kwargs...), vcat, 1:n)

"""
    ep_serial!(alg; rnd=true)

Serial EP pass. When `rnd=true` the pass goes over the data in a
random order. 
"""
function ep_serial!(alg; rnd=true) 
    rnge = rnd ? shuffle(1:length(alg.data)) : 1:length(alg.data)
    iter = ProgressBar(rnge)
    nacc = n = 0
    ev = -Inf
    trace = map(iter) do i
        desc = string(@sprintf("%8.3gh%4d%4d/%6d%11.3f", alg.h, i, nacc, n, ev))
        set_description(iter, desc)
        nacc, n, h, full, cavity, _ = ep_iteration(alg, i)
        #alg.h = h
        update!(alg, full, cavity, i, nacc/n) 
        ev = evidence(alg)
        alg.model, ev
    end 
    prune!(alg)
    return trace
end

# a helper struct for storing the EP inner iteration simulations
mutable struct Simulations{S,T}
    accsims::Vector{Tuple{T,S}}
    allsims::Vector{Tuple{T,S}}
    us::Vector{T}
end

function Simulations(S)
    accsims = Tuple{Float64,S}[]
    allsims = Tuple{Float64,S}[]
    Simulations(accsims, allsims, Float64[])
end

idinit(tree) = Dict(id(n)=>[id(n)] for n in getleaves(tree))
    
"""
    ep_iteration(alg, i)

Do a single EP site update.
"""
function ep_iteration(alg, i)
    @unpack model, sites, λ, α, minacc, target, maxsim, tuneh = alg
    X = alg.data[i]
    cavity = getcavity(i, model, sites)
    smpler = MSCSampler(cavity)
    sptree = randtree(smpler)
    init = idinit(sptree)
    sims = Simulations(eltype(smpler))
    if alg.batch > 1 && Threads.nthreads() > 1
        n, nacc = _inner_threaded!(X, smpler, init, alg, sims)
    else
        n, nacc = _inner_serial!(X, smpler, init, alg, sims)
    end
    h, nacc = _tuneh!(alg, n, nacc, sims)
    full, nacc = _momentmatching(alg, cavity, nacc, sims)
    return nacc, n, h, full, cavity, sims
end

"""
    _inner_serial!(...)

This is in the ABC simulation algorithm within the EP site update (see Bartelmé
& Chopin), implemented in a serial fashion.
"""
function _inner_serial!(X, smpler, init, alg, sims)
    nacc = n = 0
    corr = log(alg.h) 
    while true   # this could be parallelized to some extent using blocks
        n += 1
        sptree = randtree(smpler)
        G = randsplits(MSC(sptree, init))
        l = logpdf(X, G)
        u = log(rand())
        if u - corr < l
            nacc += 1
            push!(sims.accsims, (l, sptree))
        end
        push!(sims.us, u)  # store the random numbers...
        push!(sims.allsims, (l, sptree))
        (n ≥ alg.maxsim || nacc ≥ alg.target) && break
        sptree = randtree(smpler)
    end
    return n, nacc
end

"""
    _inner_threaded!(...)

See `_inner_serial!`, but in a multi-threaded implementation using a
user-defined batch size for simulation batches executed in parallel.
"""
function _inner_threaded!(X, smpler, init, alg, sims)
    nacc = n = 0
    corr = log(alg.h) 
    while true
        # simulate in parallel in batches of size `alg.batch`
        y = similar(sims.allsims, alg.batch)
        Threads.@threads for j=1:alg.batch
            sptree = randtree(smpler)
            G = randsplits(MSC(sptree, init))
            l = logpdf(X, G)
            y[j] = (l, sptree)
        end
        u = log.(rand(alg.batch))
        accepted = u .- corr .< first.(y)
        sims.accsims = vcat(sims.accsims, y[accepted])
        sims.allsims = vcat(sims.allsims, y)
        sims.us = vcat(sims.us, u)
        nacc += sum(accepted)
        n += alg.batch
        (n ≥ alg.maxsim || nacc ≥ alg.target) && break
    end
    return n, nacc
end

"""
    _tuneh!(...)

Tune the `h` parameter and re-evaluate accepted simulations.
"""
#function _tuneh!(alg, n, nacc, sims)
#    @unpack us, allsims, accsims = sims
#    !alg.tuneh && return alg.h, nacc
#    !(nacc < alg.target || alg.h != 1.) && return alg.h, nacc
#    ls = first.(allsims)
#    Ep = exp(logsumexp(ls))/n  # expected number of accepted simulations
#    h = max(1., alg.target / (alg.maxsim * Ep))
#    # if we did not reach the target, the new r is better for the
#    # current set of simulations.
#    us .-= log(h)  # re-use the random numbers but with different `r`
#    ix = filter(i->us[i] < allsims[i][1], 1:length(allsims))
#    sims.accsims = allsims[ix]
#    nacc = length(sims.accsims)
#    return h, nacc
#end
function _tuneh!(alg, n, nacc, sims)
    @unpack us, allsims, accsims = sims
    !alg.tuneh && return alg.h, nacc
#    !(nacc < alg.target || alg.h != 1.) && return alg.h, nacc
    ls = first.(allsims)
    Ep = exp(logsumexp(ls))/n
    h = min(alg.h, max(1., alg.target / (alg.maxsim * Ep)))
    alg.h = (1-alg.ν) * alg.h + alg.ν * h  # convex update
    alg.h, nacc
end
    
"""
    _momentmatching(...)

Compute the new global approximation by approximating the tilted distribution
within the exponential family using some form of moment matching.
"""
function _momentmatching(alg, cavity, nacc, sims)
    nacc < alg.minacc && return alg.model, 0
    full = matchmoments(last.(sims.accsims), cavity, alg.α)
    return full, nacc
end

#"""
#    ep_parallel!(alg)
#
#Embarrassingly parallel EP pass.
#"""
#function ep_parallel!(alg)
#    N = length(alg.data)
#    iter = ProgressBar(1:N)
#    Threads.@threads for i in iter
#        nacc, n, h, full, cavity, _ = ep_iteration(alg, i)
#        alg.sites[i] = full - cavity
#        desc = string(@sprintf("%8.3gh%4d%4d/%6d", h, i, nacc, n))
#        set_description(iter, desc)
#    end
#    alg.sites[1:N] .= map(x->alg.λ*x, alg.sites[1:N])
#    alg.model = reduce(+, alg.sites)
#    # XXX deal with damped updates... 
#    prune!(alg)
#    return alg.model
#end
    
function update!(alg, full, cavity, i, Z)
    @unpack λ, prunetol = alg
    siteup = λ * (full - alg.model)
    prunetol != 0. && prune!(siteup, prunetol)  # should we?
    alg.sites[i] = isassigned(alg.sites, i) ? alg.sites[i] + siteup : siteup
    alg.model = alg.model + siteup
    if Z > 0
        alg.siteC[i] = log(Z) - logpartition(alg.model) + logpartition(cavity)
    end
    # I am not 100% sure about the normalizing constant -- how does this play
    # with λ? should we use the updated model or not?
end

function prune!(alg)
    @unpack prunetol, sites = alg
    prunetol == 0. && return
    map(i->isassigned(sites, i) && prune!(sites[i], prunetol), 1:length(sites))
    alg.model = reduce(+, sites)
end

"""
    traceback
"""
function traceback(trace; sigdigits=3)
    clades = keys(trace[end].S.smap)
    splits = Dict(γ=>collect(keys(trace[end].S.smap[γ].splits)) for γ in clades)
    qclade = keys(trace[end].q.cmap)
    traces = Dict(γ=>Vector{Float64}[] for γ in clades)
    θtrace = Dict(γ=>Vector{Float64}[] for γ in qclade)
    for i=length(trace):-1:1
        m = SmoothTree.MomMBM(trace[i].S)
        q = trace[i].q
        for γ in clades
            x = map(δ->haskey(m, γ) ? m[γ][δ] : NaN, splits[γ])
            push!(traces[γ], x)
        end
        for γ in qclade
            y = gaussian_nat2mom(q[γ])
            push!(θtrace[γ], y)
        end
    end
    f(x) = round.(permutedims(hcat(reverse(x)...)), sigdigits=sigdigits)
    c = Dict(γ=>f(xs) for (γ, xs) in traces)
    θ = Dict(γ=>f(xs) for (γ, xs) in θtrace)
    return c, θ
end

evidence(alg) = sum(alg.siteC) + logpartition(alg.model) - logpartition(prior(alg))

# ad hoc regularized estimate of the marginal likelihood (imputes -Inf site
# norm factors as having value ϵ (should be < log(1/M) at least).
function evidence(alg, ϵ)
    Z = logpartition(alg.model) - logpartition(prior(alg))
    for Ci in alg.siteC
        Z += isfinite(Ci) ? Ci : ϵ
    end
    return Z
end


