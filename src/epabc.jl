# We approximate the tree topology posterior by a BMP
# We approximate the branch parameter posterior by independent
# Gaussians, one for each *clade*, i.e. representing the parameter for
# the branch leading to that clade as a crown group.

# We will need a struct representing a single site model, and a struct
# representing the overall approximation. The model object consists of
# some algoithm settings, the overall approximation and the sites of
# the approximation (as a vector, one for each data point).

abstract type AbstractEPABC end
Base.show(io::IO, alg::AbstractEPABC) = write(io, "$(typeof(alg))")

"""
    MSCModel

An object for conducting variational species tree inference under the
multispecies coalescent model.
"""
struct MSCModel{T,V,W}
    S::NatBMP{T,V}       # species tree distribution approximation
    q::BranchModel{T,W}  # branch parameter distribution approximation
    m::BiMap{T,String}   # species label to clade map
end

Base.show(io::IO, m::MSCModel) = write(io, "$(typeof(m))")

# initialize a MSCModel
MSCModel(x::NatBMP, θprior, m) = MSCModel(x, BranchModel(x, θprior), m)

"""
    getcavity(full, site)

Get the cavity distribution by 'taking out' `site` from `full`.
"""
function getcavity(full::MSCModel{T,V,W}, site) where {T,V,W}
    b = cavity(full.q, site.q)
    S = cavity(full.S, site.S)
    return MSCModel(S, b, full.m)
end

"""
    randsptree(model::MSCModel)

Simulate a species tree from an MSCModel (in the EP-ABC algorithm this
is used to simulate from the cavity)
"""
function randsptree(model::MSCModel)
    S = randtree(model.S)  # a random tree topology
    # now get branch parameters for the relevant clades
    # XXX need to deal with the rooted case as well (another Inf
    # length branch)
    _randbranches!(S, model.q)
    return S
end

"""
    updated_model(accepted_trees, model, cavity, alg)

Method to update the full approximation by moment matching.  This
matches the moments of the BMP distribution to the Dirichlet-BMP
posterior for the accepted trees and will compute the new full
approximation using a so-called damped update (a convex combination of
two natural parameter vectors).
"""
function updated_model(accepted_trees, model, cavity, alg)
    T = typeof(alg.model.S.root)
    m = taxonmap(accepted_trees[1], T)  # XXX sucks?
    M = NatBMP(CCD(accepted_trees, lmap=m, α=alg.α))#, αroot=0.)) respect rooting?
    q = approximate_tilted(accepted_trees, cavity.q)
    S = convexcombination(M, model.S, alg.λ)
    q = convexcombination(q, model.q, alg.λ)
    S = prune(S, atol=1e-9)
    # XXX prune q as well?
    MSCModel(S, q, model.m)
end

new_site(new_full, cavity) = getcavity(new_full, cavity)

# the main EP struct
mutable struct EPABC{X,M} <: AbstractEPABC
    data ::X
    model::M
    sites::Vector{M}
    λ    ::Float64  # for damped update...
    α    ::Float64  # Dirichlet-BMP parameter for 'moment matching'
end

function EPABC(data, model::T; λ=1., α=0.1) where T
    sites = Vector{T}(undef, length(data))
    EPABC(data, model, sites, λ, α)
end

"""
    ep_iteration!(alg, i; kwargs...)

Do an EP-ABC update for data point i, conducting simulations until we
get `target` accepted simulations or exceed a total of `maxn`
simulations. If the number of accepted draws is smaller than `mina`
the update failed, unless `fillup=true`, in which case the top `n`
simulations are added to the accepted replicates until `mina`
simulation replicates are obtained.
"""
function ep_iteration!(alg, i; mina=10, target=100, maxn=1e5,
                       noisy=false, fillup=false)
    @unpack data, model, sites = alg
    x = data[i]
    cavity = isassigned(sites, i) ? getcavity(model, sites[i]) : model
    S = randsptree(cavity)
    init = Dict(id(n)=>[id(n)] for n in getleaves(S))
    # XXX the init is where the gene to species mapping happens!
    sims = Tuple{Float64,typeof(S)}[]
    accepted = Tuple{Float64,typeof(S)}[]
    nacc = n = 0
    while true   # this could be parallelized to some extent using blocks
        n += 1
        G = randsplits(MSC(S, init))
        l = logpdf(x, G)
        noisy && n % 1000 == 0 && (@info "$n $l")
        if log(rand()) < l
            nacc += 1
            noisy && (@info "accepted! $l ($nacc)")
            push!(accepted, (l, S))
        else
            push!(sims, (l, S))
        end
        (n ≥ maxn || nacc ≥ target) && break
        S = randsptree(cavity)
    end
    if nacc < mina && !fillup
        return false, nacc, n, alg.model, cavity
    elseif nacc < mina  # fill up
        top = sort(sims, by=first, rev=true)[1:(mina-nacc)]
        noisy && (@info "added top $(mina-nacc), <ℓ>=$(mean(first.(top)))")
        push!(accepted, top...)
    end
    acc_S = last.(accepted)
    acc_l = first.(accepted)
    model_ = updated_model(acc_S, model, cavity, alg)
    site_  = new_site(model_, cavity)
    return true, nacc, n, model_, site_
end
#function ep_iteration!(alg, i; mina=10, target=100, maxn=1e5, noisy=false, adhoc=0.)
#    @unpack data, model, sites = alg
#    x = data[i]
#    cavity = isassigned(sites, i) ? getcavity(model, sites[i]) : model
#    S = randsptree(cavity)
#    init = Dict(id(n)=>[id(n)] for n in getleaves(S))
#    # XXX the init is where the gene to species mapping happens!
#    accepted = typeof(S)[]
#    nacc = n = 0
#    while true   # this could be parallelized to some extent using blocks
#        n += 1
#        G = randsplits(MSC(S, init))
#        l = logpdf(x, G) + adhoc
#        noisy && n % 1000 == 0 && (@info "$n $l")
#        if log(rand()) < l
#            noisy && (@info "accepted! ($nacc)")
#            push!(accepted, S)
#            nacc += 1
#        end
#        (n ≥ maxn || nacc ≥ target) && break
#        S = randsptree(cavity)
#    end
#    nacc < mina && return false, nacc, n, alg.model, cavity
#    model_ = updated_model(accepted, model, cavity, alg)
#    site_  = new_site(model_, cavity)
#    return true, nacc, n, model_, site_
#end

"""
    ep_pass!(alg; k=1, kwargs...)

Do a full serial EP pass over the data.
"""
function ep_pass!(alg; k=1, rnd=true, kwargs...) 
    rnge = rnd ? shuffle(1:length(alg.data)) : 1:length(alg.data)
    iter = ProgressBar(rnge)
    nacc = n = 0
    trace = map(iter) do i
        set_description(iter, string(@sprintf("pass%2d%4d%4d/%6d", k, i, nacc, n)))
        accepted, nacc, n, model, site = ep_iteration!(alg, i; kwargs...)
        if accepted
            alg.sites[i] = site
            alg.model = model
        end
        model
    end 
end

"""
    ep!(alg, n=1; kwargs...)

Do n EP passes.
"""
ep!(alg, n=1; kwargs...) = mapreduce(i->ep_pass!(alg; k=i, kwargs...), vcat, 1:n)

# trace back to analyze the EP approximation
function traceback(trace)
    clades = keys(trace[end].S.smap)
    splits = Dict(γ=>collect(keys(trace[end].S.smap[γ].splits)) for γ in clades)
    traces = Dict(γ=>Vector{Float64}[] for γ in clades)
    θtrace = Dict(γ=>Vector{Float64}[] for γ in clades)
    for i=length(trace):-1:1
        bmp = SmoothTree.MomBMP(trace[i].S)
        q = trace[i].q
        for γ in clades
            x = map(δ->haskey(bmp, γ) ? bmp[γ][δ] : NaN, splits[γ])
            y = [gaussian_nat2mom(q[γ]...)...]
            push!(traces[γ], x)
            push!(θtrace[γ], y)
        end
    end
    c = Dict(γ=>permutedims(hcat(reverse(xs)...)) for (γ, xs) in traces)
    θ = Dict(γ=>permutedims(hcat(reverse(xs)...)) for (γ, xs) in θtrace)
    return c, θ
end


# parallelizable
function pep_iteration!(alg, i; mina=10, target=100, maxn=1e5, fillup=false)
    @unpack data, model, sites = alg
    cavity = isassigned(sites, i) ? getcavity(model, sites[i]) : model
    sptree = randsptree(cavity)
    # NOTE: the init is where the gene to species mapping happens!
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
        (n ≥ maxn || nacc ≥ target) && break
        sptree = randsptree(cavity)
    end
    if nacc < mina  # fill up
        top = sort(othsims, by=first, rev=true)[1:(mina-nacc)]
        push!(accsims, top...)
    end
    acc_S = last.(accsims)
    acc_l = first.(accsims)
    model_ = updated_model(acc_S, model, cavity, alg)
    site_  = new_site(model_, cavity)
    return nacc, n, site_
end

# parallel EP pass
function pep_pass!(alg; k=1, kwargs...)
    iter = ProgressBar(1:length(alg.data))
    Threads.@threads for i in iter
        nacc, n, site = pep_iteration!(alg, i; kwargs...) 
        alg.sites[i] = site
        set_description(iter, string(@sprintf("pass%2d%4d%4d/%6d", k, i, nacc, n)))
    end
end

