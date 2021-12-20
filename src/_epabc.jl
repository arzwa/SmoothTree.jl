# We approximate the tree topology posterior by a BMP
# We approximate the branch parameter posterior by independent
# Gaussians, one for each *clade*, i.e. representing the parameter for
# the branch leading to that clade as a crown group.
#
# We will need a struct representing a single site model, and a struct
# representing the overall approximation. The model object consists of
# some algoithm settings, the overall approximation and the sites of
# the approximation (as a vector, one for each data point).

# Hold for each clade, potentially, the natural parameters of a
# Gaussian, but only store explicitly when distinct from the prior.
struct BranchModel{T,V}
    cmap ::Dict{T,V}  # clade => natural parameter 
    prior::V  # natural parameter for prior
end

# initialize an empty branchmodel object
BranchModel(x::CCD{T}, prior) where T = BranchModel(Dict{T,Vector{Float64}}(), prior)

struct MSCModel{T,V,W}
    Ψ::CCD{T,V}  # species tree distribution approximation
    q::BranchModel{T,W}  # branch parameter distribution approximation
end

# initialize a MSCModel
MSCModel(x::CCD, θprior) = MSCModel(x, BranchModel(x, θprior))

# the main EP struct
struct EPABC{X,M}
    data ::X
    model::M
    sites::Vector{M}
    M    ::Int
    λ    ::Float64  # for fractional update...
end

function EPABC(data, model::T; M=10000, λ=1.) where T
    leafset = collect(keys(model.Ψ.lmap))
    initsite = MSCModel(CCD(leafset), model.q)
    sites = [deepcopy(initsite) for i=1:length(data)]
    EPABC(data, model, sites, M, λ)
end

# 1. We need a method to form the cavity density
# it should not be a problem that this modifies the full approximation
function getcavity!(full, site)
    @unpack Ψ, q = full
    for (clade, η) in site.q.cmap
        q[clade] -= η
    end
    for (clade, count) in site.Ψ.cmap
        Ψ.cmap[clade] -= count
        isleafclade(clade) && continue
        for (split, count) in site.Ψ.smap[clade]
            Ψ.smap[clade][split] -= count
        end
    end
    return full
end

# 2. We need a method to simulate from the cavity, basically a method
# to simulate MSC models from the MSCModel or whatever we call it
function randsptree(model::MSCModel)
    S = randtree(model.Ψ)  # a random tree topology
    # now get branch parameters for the relevant clades
    # XXX need to deal with the rooted case as well (another Inf
    # length branch)
    _randbranches!(S, model.q, model.Ψ.lmap)
    return S
end

function _randbranches!(node, q, lmap)
    if isleaf(node)
        node.data.distance == Inf
        return lmap[name(node)]
    else
        left = _randbranches!(node[1], q, lmap)
        rght = _randbranches!(node[2], q, lmap)
        clade = left + rght
        if haskey(q.cmap, clade)
            η = q.cmap[clade]
        else
            η = q.prior
        end
        node.data.distance = exp(rand(gaussian_nat(η...)))
        return clade
    end
end

# get a univariate gaussian from natural parameters
function gaussian_nat(η1, η2)
    μ = -η1/(2η2)
    σ = √(-1/(2η2))
    Normal(μ, σ)
end

# 3. We need a method to update the full approximation by moment
# matching. We need something like the BranchModel storing for each
# clade in the accepted simulations the count for that clade, and its
# mean and squared parameter value
function updated_model(accepted_trees, model, n)
    Ψ = CCD(accepted_trees, α=model.Ψ.α)
    # how do we actually get an updated CCD? will just doing the above
    # work? Whatabout α?
    branchparams = Dict(clade=>zeros(3) for clade in keys(Ψ.smap))
    for tree in accepted_trees
        _record_branchparams!(branchparams, tree, Ψ.lmap)
    end
    q = BranchModel(moment2natural(branchparams), model.q.prior)
    MSCModel(Ψ, q)
end

function _record_branchparams!(d, node, lmap)
    isleaf(node) && return lmap[name(node)]
    left = _record_branchparams!(d, node[1], lmap) 
    rght = _record_branchparams!(d, node[2], lmap) 
    clade = left + rght
    x = log(node.data.distance)
    d[clade] .+= [1., x, x^2]
    return clade
end

function moment2natural(branchparams)
    d = Dict(clade => moment2natural(x...) for (clade, x) in branchparams)
    return d
end

moment2natural(n, xs, xsqs) = moment2natural(xs/n, xsqs/n)
moment2natural(μ, V) = [μ/V, -1.0/(2V)]

function new_site(new_full, cavity)
    getcavity!(deepcopy(new_full), cavity)
end

function ep_iteration!(alg, i)
    @unpack data, model, sites, M = alg
    x = data[i]
    site = sites[i]
    cavity = getcavity!(model, site)
    simulations = map(1:M) do m
        S = randsptree(cavity)
        G = randsplits(MSC(S))
        l = logpdf(x, G) 
        accept = log(rand()) < l
        accept, l, S
    end
    accepted = filter(first, simulations)
    nacc = length(accepted)
    @show nacc
    model_ = updated_model(last.(accepted), model)
    return model_, cavity
    alg.sites[i] = new_site(model_, cavity) 
    alg.model = model_
end





