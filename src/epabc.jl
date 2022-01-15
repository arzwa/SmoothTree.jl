abstract type AbstractEPABC end

# the main EP struct
@with_kw mutable struct EPABC{X,M} <: AbstractEPABC
    data ::X
    model::M
    sites::Vector{M}
    λ::Float64 = 0.1    # for damped update...
    α::Float64 = 1e-16  # Dirichlet-MBM parameter for 'moment matching'
    minacc = 100
    target = 500
    maxsim = 1e5 
    fillup = true
    prunetol = 0.
end

function EPABC(data, prior::T; kwargs...) where T
    sites = Vector{T}(undef, length(data)+1)
    sites[end] = prior  # last entry is the prior
    EPABC(data=data, model=prior, sites=sites; kwargs...)
end

function prune!(alg)
    alg.prunetol == 0. && return
    map(site->prune!(site, alg.prunetol), alg.sites)
    alg.model = reduce(+, alg.sites)
end

# get the cavity distribution
getcavity(i, m, s) = isassigned(s, i) ? m - s[i] : m

function ep_iteration(alg, i)
    @unpack data, model, sites, λ, α = alg
    @unpack minacc, target, maxsim, fillup = alg
    X = alg.data[i]
    cavity = getcavity(i, model, sites)
    smpler = MSCSampler(cavity)
    sptree = randtree(smpler)
    init = Dict(id(n)=>[id(n)] for n in getleaves(sptree))
    S = typeof(sptree)
    accsims = Tuple{Float64,S}[]
    othsims = Tuple{Float64,S}[]
    nacc = n = 0
    while true   # this could be parallelized to some extent using blocks
        n += 1
        G = randsplits(MSC(sptree, init))
        l = logpdf(data[i], G)
        if log(rand()) < l
            nacc += 1
            push!(accsims, (l, sptree))
        else
            push!(othsims, (l, sptree))
        end
        (n ≥ maxsim || nacc ≥ target) && break
        sptree = randtree(smpler)
    end
    if (nacc < minacc) && !fillup  # failed update
        full = model
    else
        top = sort(othsims, by=first, rev=true)[1:(minacc-nacc)]
        push!(accsims, top...)
        accS = last.(accsims)
        full = matchmoments(accS, cavity, alg.α)
    end
    return nacc, n, full, cavity, accsims
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
        set_description(iter, string(@sprintf("%4d%4d/%6d", i, nacc, n)))
        nacc, n, full, cavity, _ = ep_iteration(alg, i)
        update!(alg, full, cavity, i) 
    end 
    prune!(alg)
    return alg.model
end

"""
    ep_parallel!(alg)

Embarrassingly parallel EP pass.
"""
function ep_parallel!(alg)
    N = length(alg.data)
    iter = ProgressBar(1:N)
    Threads.@threads for i in iter
        nacc, n, full, cavity, _ = ep_iteration(alg, i)
        alg.sites[i] = full - cavity
        set_description(iter, string(@sprintf("%4d%4d/%6d", i, nacc, n)))
    end
    alg.sites[1:N] .= map(x->alg.λ*x, alg.sites[1:N])
    alg.model = reduce(+, alg.sites)
    # XXX deal with damped updates... 
    prune!(alg)
    return alg.model
end
    
# not using the cavity...
function update!(alg, full, cavity, i)
    @unpack λ = alg
    siteup = λ * (full - alg.model)
    alg.sites[i] = isassigned(alg.sites, i) ? 
        alg.sites[i] + siteup : siteup
    alg.model = alg.model + siteup
end

"""
    traceback
"""
function traceback(trace)
    clades = keys(trace[end].S.smap)
    splits = Dict(γ=>collect(keys(trace[end].S.smap[γ].splits)) for γ in clades)
    traces = Dict(γ=>Vector{Float64}[] for γ in clades)
    θtrace = Dict(γ=>Vector{Float64}[] for γ in clades)
    for i=length(trace):-1:1
        bmp = SmoothTree.MomMBM(trace[i].S)
        q = trace[i].q
        for γ in clades
            x = map(δ->haskey(bmp, γ) ? bmp[γ][δ] : NaN, splits[γ])
            y = gaussian_nat2mom(q[γ])
            push!(traces[γ], x)
            push!(θtrace[γ], y)
        end
    end
    c = Dict(γ=>permutedims(hcat(reverse(xs)...)) for (γ, xs) in traces)
    θ = Dict(γ=>permutedims(hcat(reverse(xs)...)) for (γ, xs) in θtrace)
    return c, θ
end


#"""
#    ep_iteration!(alg, i; kwargs...)
#
#Do an EP-ABC update for data point i, conducting simulations until we
#get `target` accepted simulations or exceed a total of `maxn`
#simulations. If the number of accepted draws is smaller than `mina`
#the update failed, unless `fillup=true`, in which case the top `n`
#simulations are added to the accepted replicates until `mina`
#simulation replicates are obtained.
#"""
#function ep_iteration!(alg, i; mina=10, target=100, maxn=1e5,
#                       noisy=false, fillup=false)
#    @unpack data, model, sites = alg
#    x = data[i]
#    cavity = isassigned(sites, i) ? getcavity(model, sites[i]) : model
#    S = randsptree(cavity)
#    init = Dict(id(n)=>[id(n)] for n in getleaves(S))
#    # XXX the init is where the gene to species mapping happens!
#    sims = Tuple{Float64,typeof(S)}[]
#    accepted = Tuple{Float64,typeof(S)}[]
#    nacc = n = 0
#    while true   # this could be parallelized to some extent using blocks
#        n += 1
#        G = randsplits(MSC(S, init))
#        l = logpdf(x, G)
#        noisy && n % 1000 == 0 && (@info "$n $l")
#        if log(rand()) < l
#            nacc += 1
#            noisy && (@info "accepted! $l ($nacc)")
#            push!(accepted, (l, S))
#        else
#            push!(sims, (l, S))
#        end
#        (n ≥ maxn || nacc ≥ target) && break
#        S = randsptree(cavity)
#    end
#    if nacc < mina && !fillup
#        return false, nacc, n, alg.model, cavity
#    elseif nacc < mina  # fill up
#        top = sort(sims, by=first, rev=true)[1:(mina-nacc)]
#        noisy && (@info "added top $(mina-nacc), <ℓ>=$(mean(first.(top)))")
#        push!(accepted, top...)
#    end
#    acc_S = last.(accepted)
#    acc_l = first.(accepted)
#    model_ = updated_model(acc_S, model, cavity, alg)
#    site_  = new_site(model_, cavity)
#    return true, nacc, n, model_, site_
#end
##function ep_iteration!(alg, i; mina=10, target=100, maxn=1e5, noisy=false, adhoc=0.)
##    @unpack data, model, sites = alg
##    x = data[i]
##    cavity = isassigned(sites, i) ? getcavity(model, sites[i]) : model
##    S = randsptree(cavity)
##    init = Dict(id(n)=>[id(n)] for n in getleaves(S))
##    # XXX the init is where the gene to species mapping happens!
##    accepted = typeof(S)[]
##    nacc = n = 0
##    while true   # this could be parallelized to some extent using blocks
##        n += 1
##        G = randsplits(MSC(S, init))
##        l = logpdf(x, G) + adhoc
##        noisy && n % 1000 == 0 && (@info "$n $l")
##        if log(rand()) < l
##            noisy && (@info "accepted! ($nacc)")
##            push!(accepted, S)
##            nacc += 1
##        end
##        (n ≥ maxn || nacc ≥ target) && break
##        S = randsptree(cavity)
##    end
##    nacc < mina && return false, nacc, n, alg.model, cavity
##    model_ = updated_model(accepted, model, cavity, alg)
##    site_  = new_site(model_, cavity)
##    return true, nacc, n, model_, site_
##end
#
#"""
#    ep_pass!(alg; k=1, kwargs...)
#
#Do a full serial EP pass over the data.
#"""
#function ep_pass!(alg; k=1, rnd=true, kwargs...) 
#    rnge = rnd ? shuffle(1:length(alg.data)) : 1:length(alg.data)
#    iter = ProgressBar(rnge)
#    nacc = n = 0
#    trace = map(iter) do i
#        set_description(iter, string(@sprintf("pass%2d%4d%4d/%6d", k, i, nacc, n)))
#        accepted, nacc, n, model, site = ep_iteration!(alg, i; kwargs...)
#        if accepted
#            alg.sites[i] = site
#            alg.model = model
#        end
#        model
#    end 
#end
#
#"""
#    ep!(alg, n=1; kwargs...)
#
#Do n EP passes.
#"""
#ep!(alg, n=1; kwargs...) = mapreduce(i->ep_pass!(alg; k=i, kwargs...), vcat, 1:n)
#
## trace back to analyze the EP approximation
#function traceback(trace)
#    clades = keys(trace[end].S.smap)
#    splits = Dict(γ=>collect(keys(trace[end].S.smap[γ].splits)) for γ in clades)
#    traces = Dict(γ=>Vector{Float64}[] for γ in clades)
#    θtrace = Dict(γ=>Vector{Float64}[] for γ in clades)
#    for i=length(trace):-1:1
#        bmp = SmoothTree.MomMBM(trace[i].S)
#        q = trace[i].q
#        for γ in clades
#            x = map(δ->haskey(bmp, γ) ? bmp[γ][δ] : NaN, splits[γ])
#            y = [gaussian_nat2mom(q[γ]...)...]
#            push!(traces[γ], x)
#            push!(θtrace[γ], y)
#        end
#    end
#    c = Dict(γ=>permutedims(hcat(reverse(xs)...)) for (γ, xs) in traces)
#    θ = Dict(γ=>permutedims(hcat(reverse(xs)...)) for (γ, xs) in θtrace)
#    return c, θ
#end
#
#
## parallelizable
#function pep_iteration!(alg, i; mina=10, target=100, maxn=1e5, fillup=false)
#    @unpack data, model, sites = alg
#    cavity = isassigned(sites, i) ? getcavity(model, sites[i]) : model
#    sptree = randsptree(cavity)
#    # NOTE: the init is where the gene to species mapping happens!
#    init = Dict(id(n)=>[id(n)] for n in getleaves(sptree))
#    S = typeof(sptree)
#    accsims = Tuple{Float64,S}[]
#    othsims = Tuple{Float64,S}[]
#    nacc = n = 0
#    while true   # this could be parallelized to some extent using blocks
#        n += 1
#        G = randsplits(MSC(sptree, init))
#        l = logpdf(data[i], G)
#        if log(rand()) < l
#            nacc += 1
#            push!(accsims, (l, sptree))
#        else
#            push!(othsims, (l, sptree))
#        end
#        (n ≥ maxn || nacc ≥ target) && break
#        sptree = randsptree(cavity)
#    end
#    if nacc < mina  # fill up
#        top = sort(othsims, by=first, rev=true)[1:(mina-nacc)]
#        push!(accsims, top...)
#    end
#    acc_S = last.(accsims)
#    acc_l = first.(accsims)
#    model_ = updated_model(acc_S, model, cavity, alg)
#    site_  = new_site(model_, cavity)
#    return nacc, n, site_
#end
#
## parallel EP pass
#function pep_pass!(alg; k=1, kwargs...)
#    iter = ProgressBar(1:length(alg.data))
#    Threads.@threads for i in iter
#        nacc, n, site = pep_iteration!(alg, i; kwargs...) 
#        alg.sites[i] = site
#        set_description(iter, string(@sprintf("pass%2d%4d%4d/%6d", k, i, nacc, n)))
#    end
#end

