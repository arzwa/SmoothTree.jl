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
    λ::Float64 = 0.1    # for damped update...
    α::Float64 = 1e-16  # Dirichlet-MBM parameter for 'moment matching'
    minacc::Int = 100
    target::Int = 500
    maxsim::Int = 1e5 
    h::Float64 = 1.
    tuneh::Bool = true
    batch::Int = 1
    prunetol::Float64 = 0.
end

function tuneoff!(alg)
    alg.h = 1.
    alg.tuneh = false
end

function EPABC(data, prior::T; kwargs...) where T
    sites = Vector{T}(undef, length(data)+1)
    sites[end] = prior  # last entry is the prior
    EPABC(data=data, model=prior, sites=sites; kwargs...)
end

# get the cavity distribution
getcavity(i, m, s) = isassigned(s, i) ? m - s[i] : m

function ep_iteration(alg, i)
    @unpack model, sites, λ, α, minacc, target, maxsim, tuneh = alg
    X = alg.data[i]
    cavity = getcavity(i, model, sites)
    smpler = MSCSampler(cavity)
    sptree = randtree(smpler)
    init = Dict(id(n)=>[id(n)] for n in getleaves(sptree))
    S = typeof(sptree)
    accsims = Tuple{Float64,S}[]
    allsims = Tuple{Float64,S}[]
    us = Float64[]
    nacc = n = 0
    corr = log(alg.h) 
    while true   # this could be parallelized to some extent using blocks
        n += 1
        G = randsplits(MSC(sptree, init))
        l = logpdf(X, G)
        u = log(rand())
        if u - corr < l
            nacc += 1
            push!(accsims, (l, sptree))
        end
        push!(us, u)  # store the random numbers...
        push!(allsims, (l, sptree))
        (n ≥ maxsim || nacc ≥ target) && break
        sptree = randtree(smpler)
    end
    h = alg.h
    if tuneh # we are tuning h
        ls = first.(allsims)
        Ep = exp(logsumexp(ls))/n  # expected number of accepted simulations
        if nacc < target || h != 1.
            h = max(1., target / (maxsim * Ep))
            # if we did not reach the target, the new r is better for the
            # current set of simulations.
            us .-= log(h)  # re-use the random numbers but with different `r`
            ix = filter(i->us[i] < allsims[i][1], 1:length(allsims))
            accsims = allsims[ix]
            nacc = length(accsims)
        end
    end
    full = nacc < minacc ? # not enough accepted simulations
        model : matchmoments(last.(accsims), cavity, alg.α)
    return nacc, n, h, full, cavity, accsims
end

# simulation-parallel iteration
function pep_iteration(alg, i)
    @unpack model, sites, λ, α, minacc, target, maxsim, tuneh = alg
    X = alg.data[i]
    cavity = getcavity(i, model, sites)
    smpler = MSCSampler(cavity)
    sptree = randtree(smpler)
    init = Dict(id(n)=>[id(n)] for n in getleaves(sptree))
    S = typeof(sptree)
    accsims = Tuple{Float64,S}[]
    allsims = Tuple{Float64,S}[]
    us = Float64[]
    nacc = n = 0
    corr = log(alg.h) 
    while true
        # simulate in parallel in batches of size `alg.batch`
        sims = similar(allsims, alg.batch)
        Threads.@threads for j=1:alg.batch
            sptree = randtree(smpler)
            G = randsplits(MSC(sptree, init))
            l = logpdf(X, G)
            sims[j] = (l, sptree)
        end
        u = log.(rand(alg.batch))
        accepted = u .- corr .< first.(sims)
        accsims = vcat(accsims, sims[accepted])
        allsims = vcat(allsims, sims)
        us = vcat(us, u)
        nacc += sum(accepted)
        n += alg.batch
        (n ≥ maxsim || nacc ≥ target) && break
    end
    h = alg.h
    if tuneh # we are tuning h
        ls = first.(allsims)
        Ep = exp(logsumexp(ls))/n  # expected number of accepted simulations
        if nacc < target || h != 1.
            h = max(1., target / (maxsim * Ep))
            # if we did not reach the target, the new r is better for the
            # current set of simulations.
            us .-= log(h)  # re-use the random numbers but with different `r`
            ix = filter(i->us[i] < allsims[i][1], 1:length(allsims))
            accsims = allsims[ix]
            nacc = length(accsims)
        end
    end
    full = nacc < minacc ? # not enough accepted simulations
        model : matchmoments(last.(accsims), cavity, alg.α)
    return nacc, n, h, full, cavity, accsims
end

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
    trace = map(iter) do i
        desc = string(@sprintf("%8.3gh%4d%4d/%6d", alg.h, i, nacc, n))
        set_description(iter, desc)
        if alg.batch == 1
            nacc, n, h, full, cavity, _ = ep_iteration(alg, i)
        else
            nacc, n, h, full, cavity, _ = pep_iteration(alg, i)
        end 
        alg.h = h
        update!(alg, full, cavity, i) 
    end 
    prune!(alg)
    return trace
end

"""
    ep_parallel!(alg)

Embarrassingly parallel EP pass.
"""
function ep_parallel!(alg)
    N = length(alg.data)
    iter = ProgressBar(1:N)
    Threads.@threads for i in iter
        nacc, n, h, full, cavity, _ = ep_iteration(alg, i)
        alg.sites[i] = full - cavity
        desc = string(@sprintf("%8.3gh%4d%4d/%6d", h, i, nacc, n))
        set_description(iter, desc)
    end
    alg.sites[1:N] .= map(x->alg.λ*x, alg.sites[1:N])
    alg.model = reduce(+, alg.sites)
    # XXX deal with damped updates... 
    prune!(alg)
    return alg.model
end
    
# not using the cavity...
function update!(alg, full, cavity, i)
    @unpack λ, prunetol = alg
    siteup = λ * (full - alg.model)
    prunetol != 0. && prune!(siteup, prunetol)  # should we?
    alg.sites[i] = isassigned(alg.sites, i) ? 
        alg.sites[i] + siteup : siteup
    alg.model = alg.model + siteup
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
