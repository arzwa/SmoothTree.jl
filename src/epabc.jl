# We approximate the tree topology posterior by a BMP
# We approximate the branch parameter posterior by independent
# Gaussians, one for each *clade*, i.e. representing the parameter for
# the branch leading to that clade as a crown group.
#
# We will need a struct representing a single site model, and a struct
# representing the overall approximation. The model object consists of
# some algoithm settings, the overall approximation and the sites of
# the approximation (as a vector, one for each data point).

taxonmap(l, T=UInt16) = Dict(T(2^i)=>l[i+1] for i=0:length(l)-1)
inverse(d::Dict) = Dict(v=>k for (k,v) in d)

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
Base.getindex(m::BranchModel, γ) = m.cmap[γ]

struct MSCModel{T,V,W}
    S::NatBMP{T,V}  # species tree distribution approximation
    q::BranchModel{T,W}  # branch parameter distribution approximation
    m1::Dict{T,String}  # taxon map BMP => species tree labels
    m2::Dict{String,T}  # taxon map species tree => BMP labels
end

# initialize a MSCModel
MSCModel(x::NatBMP, θprior, m) = MSCModel(x, BranchModel(x, θprior), m, inverse(m))

# the main EP struct
mutable struct EPABC{X,M}
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

# 1. We need a method to form the cavity density
# it should not be a problem that this modifies the full approximation
function getcavity(full::MSCModel{T,V,W}, site) where {T,V,W}
    q = Dict{T,W}()
    for (clade, η) in full.q.cmap
        q[clade] = haskey(site.q, clade) ? η - site.q[clade] : η
    end 
    b = BranchModel(q, full.q.prior)
    S = cavity(full.S, site.S)
    return MSCModel(S, b, full.m1, full.m2)
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
    for n in getleaves(S)  # set leaf names
        n.data.name = model.m1[id(n)]
    end
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
gaussian_nat2mom(η1, η2) = (-η1/(2η2), √(-1.0/(2η2)))
gaussian_mom2nat(μ , V ) = (μ/V, -1.0/(2V))

# 3. We need a method to update the full approximation by moment
# matching. We need something like the BranchModel storing for each
# clade in the accepted simulations the count for that clade, and its
# mean and squared parameter value
function updated_model(accepted_trees, model, cavity, λ, α)
    M = NatBMP(CCD(accepted_trees, α=α))
    S = convexcombination(M, model.S, λ)
    q = newbranches(S, accepted_trees, model, cavity, λ)
    MSCModel(S, q, model.m1, model.m2)
end

function newbranches(S, accepted_trees, model, cavity, λ)
    d = typeof(model.q.cmap)()
    N = length(accepted_trees)
    # obtain moment estimates
    for tree in accepted_trees
        _record_branchparams!(d, tree, model.m2)
    end
    _cavity_contribution!(d, cavity.q, N)
    # update natural params of full approx by moment matching
    q′ = updated_q(d, model.q, N, λ)
    return BranchModel(q′, model.q.prior)
end

function _record_branchparams!(d, node, lmap)
    isleaf(node) && return lmap[name(node)]
    left = _record_branchparams!(d, node[1], lmap) 
    rght = _record_branchparams!(d, node[2], lmap) 
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
Do an EP-ABC update, conducting simulations until we get `target`
accepted simulations or exceed a total of `maxn` simulations. If the
number of accepted draws is smaller than `mina` the update failed.
"""
function ep_iteration!(alg, i; mina=10, target=100, maxn=1e5)
    @unpack data, model, sites = alg
    x = data[i]
    cavity = isassigned(sites, i) ? getcavity(model, sites[i]) : model
    S = randsptree(cavity)
    accepted = typeof(S)[]
    nacc = n = 0
    while true
        n += 1
        G = randsplits(MSC(S))
        l = logpdf(x, G) 
        if log(rand()) < l
            push!(accepted, S)
            nacc += 1
        end
        (n ≥ maxn || nacc ≥ target) && break
        S = randsptree(cavity)
    end
    nacc < mina && return nacc, n, alg.model
    model_ = updated_model(accepted, model, cavity, alg.λ, alg.α)
    alg.sites[i] = new_site(model_, cavity) 
    alg.model = model_
    return nacc, n, alg.model
end

function ep_pass!(alg; k=1, kwargs...) 
    iter = ProgressBar(1:length(alg.data))
    nacc = n = 0
    trace = map(iter) do i
        set_description(iter, string(@sprintf("pass%2d%4d/%6d", k, nacc, n)))
        nacc, n, model = ep_iteration!(alg, i; kwargs...)
        model
    end 
end

ep!(alg, n=1; kwargs...) = mapreduce(i->ep_pass!(alg; k=i, kwargs...), vcat, 1:n)

