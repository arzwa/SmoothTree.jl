# We approximate the tree topology posterior by a BMP
# We approximate the branch parameter posterior by independent
# Gaussians, one for each *clade*, i.e. representing the parameter for
# the branch leading to that clade as a crown group.
#
# We will need a struct representing a single site model, and a struct
# representing the overall approximation. The model object consists of
# some algoithm settings, the overall approximation and the sites of
# the approximation (as a vector, one for each data point).

abstract type AbstractEPABC end
Base.show(io::IO, alg::AbstractEPABC) = write(io, "$(typeof(alg))")

# Hold for each clade, potentially, the natural parameters of a
# Gaussian, but only store explicitly when distinct from the prior.
struct BranchModel{T,V}
    cmap ::Dict{T,V}  # clade => natural parameter 
    prior::V  # natural parameter for prior
end

# initialize an empty branchmodel object
BranchModel(x::NatBMP{T}, prior::V) where {T,V} = 
    BranchModel(Dict{T,V}(), prior)

# some accessors
Base.haskey(m::BranchModel, γ) = haskey(m.cmap, γ)
Base.getindex(m::BranchModel, γ) = haskey(m, γ) ? m.cmap[γ] : m.prior

"""
    MSCModel

An object for conducting species tree inference under the MSC.
"""
struct MSCModel{T,V,W}
    S::NatBMP{T,V}       # species tree distribution approximation
    q::BranchModel{T,W}  # branch parameter distribution approximation
    m::BiMap{T,String}   # species label to clade map
end

Base.show(io::IO, m::MSCModel) = write(io, "$(typeof(m))")

# initialize a MSCModel
MSCModel(x::NatBMP, θprior, m) = MSCModel(x, BranchModel(x, θprior), m)

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

mutable struct MULEPABC{X,M} <: AbstractEPABC
    data ::X
    model::M
    sites::Vector{M}
    λ    ::Float64  # for damped update...
    α    ::Float64  # Dirichlet-BMP parameter for 'moment matching'
    speciesmap::Dict{String,Vector{String}}  # maps species name -> subgenomes
end

function MULEPABC(data, model::T, speciesmap; λ=1., α=0.1) where T
    sites = Vector{T}(undef, length(data))
    MULEPABC(data, model, sites, λ, α, speciesmap)
end

# 1. We need a method to form the cavity density
# it should not be a problem that this modifies the full approximation
function getcavity(full::MSCModel{T,V,W}, site) where {T,V,W}
    q = Dict{T,W}()
    for (clade, η) in full.q.cmap
        q[clade] = haskey(site.q, clade) ? η - site.q[clade] : η
    end 
    b = BranchModel(q, full.q.prior)
    S = cavity(full.S, site.S)
    return MSCModel(S, b, full.m)
end

# 2. We need a method to simulate from the cavity, basically a method
# to simulate MSC models from the MSCModel or whatever we call it
# XXX needs to be fast
function randsptree(model::MSCModel)
    S = randtree(model.S)  # a random tree topology
    # now get branch parameters for the relevant clades
    # XXX need to deal with the rooted case as well (another Inf
    # length branch)
    _randbranches!(S, model.q)
    return S
end

function _randbranches!(node, q)
    if isleaf(node)
        node.data.distance == Inf
        return id(node)
    else
        left = _randbranches!(node[1], q)
        rght = _randbranches!(node[2], q)
        clade = left + rght
        η = haskey(q, clade) ? q[clade] : q.prior
        node.data.distance = exp(randgaussian_nat(η[1], η[2]))
        return clade
    end
end

function randgaussian_nat(η1, η2) 
    μ, V = gaussian_nat2mom(η1, η2)
    return μ + √(V) * randn()
end

# get a univariate gaussian from natural parameters
gaussian_nat2mom(η1, η2) = (-η1/(2η2), -1.0/(2η2))
gaussian_mom2nat(μ , V ) = (μ/V, -1.0/(2V))

# 3. We need a method to update the full approximation by moment
# matching. We need something like the BranchModel storing for each
# clade in the accepted simulations the count for that clade, and its
# mean and squared parameter value
function updated_model(accepted_trees, model, cavity, alg)
    T = typeof(alg.model.S.root)
    m = taxonmap(accepted_trees[1], T)  # XXX sucks?
    M = NatBMP(CCD(accepted_trees, lmap=m, α=alg.α)) 
    S = convexcombination(M, model.S, alg.λ)
    q = newbranches(S, accepted_trees, model, cavity, alg.λ)
    MSCModel(S, q, model.m)
end

function newbranches(S, accepted_trees, model, cavity, λ)
    d = typeof(model.q.cmap)()
    N = length(accepted_trees)
    # obtain moment estimates
    for tree in accepted_trees
        _record_branchparams!(d, tree)
    end
    # add unrepresented prior samples (do or don't?)
    _cavity_contribution!(d, cavity.q, N)
    # update natural params of full approx by moment matching
    q′ = updated_q(d, model.q, N, λ)
    return BranchModel(q′, model.q.prior)
end

function _record_branchparams!(d, node)
    isleaf(node) && return id(node) #lmap[name(node)]
    left = _record_branchparams!(d, node[1]) 
    rght = _record_branchparams!(d, node[2]) 
    clade = left + rght
    x = log(node.data.distance)
    if !haskey(d, clade)
        d[clade] = zeros(3)
    end
    d[clade] .+= [1., x, x^2]
    return clade
end

# add the cavity (pseudo-prior) contribution to the moment estimates
function _cavity_contribution!(d, q, N)
    for (γ, xs) in d
        n = N - xs[1]  # number of cavity draws to 'add'
        η = haskey(q, γ) ? q[γ] : q.prior
        μ, V = gaussian_nat2mom(η...)
        d[γ][2] += n*μ
        d[γ][3] += n*(V + μ^2)
        d[γ][1] = N
    end
end

function updated_q(d, q, N, λ)
    q′ = Dict(γ => _mom2nat(x[2], x[3], x[1]) for (γ, x) in d)
    for (γ, x) in q′ # convex combination in η space (damped update)
        y = haskey(q, γ) ? q[γ] : q.prior
        q′[γ] = ((1-λ) .* y) .+ (λ .* x)
    end
    return q′
end

function _mom2nat(xs, xsqs, N) 
    μ = xs/N
    V = xsqs/N - μ^2
    [gaussian_mom2nat(μ, V)...]
end

new_site(new_full, cavity) = getcavity(new_full, cavity)

"""
    ep_iteration!(alg, i; kwargs...)

Do an EP-ABC update for data point i, conducting simulations until we
get `target` accepted simulations or exceed a total of `maxn`
simulations. If the number of accepted draws is smaller than `mina`
the update failed.
"""
function _ep_iteration!(alg, i; mina=10, target=100, maxn=1e5, noisy=false, adhoc=0.)
    @unpack data, model, sites = alg
    x = data[i]
    cavity = isassigned(sites, i) ? getcavity(model, sites[i]) : model
    S = randsptree(cavity)
    init = Dict(id(n)=>[id(n)] for n in getleaves(S))
    # XXX the init is where the gene to species mapping happens!
    accepted = typeof(S)[]
    nacc = n = 0
    while true   # this could be parallelized to some extent using blocks
        n += 1
        G = randsplits(MSC(S, init))
        l = logpdf(x, G) + adhoc
        noisy && n % 1000 == 0 && (@info "$n $l")
        if log(rand()) < l
            noisy && (@info "accepted! ($nacc)")
            push!(accepted, S)
            nacc += 1
        end
        (n ≥ maxn || nacc ≥ target) && break
        S = randsptree(cavity)
    end
    nacc < mina && return false, nacc, n, alg.model, cavity
    model_ = updated_model(accepted, model, cavity, alg)
    site_  = new_site(model_, cavity)
    return true, nacc, n, model_, site_
end

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

# parallel EP pass
function pep_pass!(alg; k=1, kwargs...)
    # do ep in parallel
    # combine sites
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


function ep_iteration!(alg::MULEPABC, i; mina=10, target=100,
                       maxn=1e5, noisy=false, adhoc=0.)
    @unpack data, model, sites = alg
    x = data[i]
    cavity = isassigned(sites, i) ? getcavity(model, sites[i]) : model
    S = randsptree(cavity)
    # XXX the init is where the gene to species mapping happens!
    accepted = typeof(S)[]
    nacc = n = 0
    while true   # this could be parallelized to some extent using blocks
        n += 1
        init = randinit(x, model.m, alg.speciesmap)
        G = randsplits(MSC(S, init))
        l = logpdf(x, G) + adhoc
        noisy && n % 1000 == 0 && (@info "$n $l")
        if log(rand()) < l
            noisy && (@info "accepted! ($nacc)")
            push!(accepted, S)
            nacc += 1
        end
        (n ≥ maxn || nacc ≥ target) && break
        S = randsptree(cavity)
    end
    nacc < mina && return false, nacc, n, alg.model, cavity
    model_ = updated_model(accepted, model, cavity, alg)
    site_  = new_site(model_, cavity)
    return true, nacc, n, model_, site_
end

function randinit(x::CCD{T}, tmap, speciesmap) where T
    # sample subgenomes without replacement!
    urn = Dict(x=>shuffle(y) for (x,y) in speciesmap)
    init = Dict{T,Vector{T}}()
    for (γ, gene) in x.lmap
        species = _spname(gene)
        subgenome = pop!(urn[species])
        spγ = tmap[subgenome]
        init[spγ] = [γ]
    end
    return init
end

# keep top x%
function ep_iteration!(alg, i; mina=10, target=100, maxn=1e5, noisy=false, adhoc=false)
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
    if nacc < mina && !adhoc 
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
