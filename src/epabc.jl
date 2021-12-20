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
BranchModel(x::BMP{T}, prior) where T = BranchModel(Dict{T,Vector{Float64}}(), prior)

struct MSCModel{T,V,W}
    Ψ::BMP{T,V}  # species tree distribution approximation
    q::BranchModel{T,W}  # branch parameter distribution approximation
    m1::Dict{T,String}  # taxon map BMP => species tree labels
    m2::Dict{String,T}  # taxon map BMP => species tree labels
end

inverse(d) = Dict(v=>k for (k,v) in d)

# initialize a MSCModel
MSCModel(x::BMP, θprior, m) = MSCModel(x, BranchModel(x, θprior), m, inverse(m))

# the main EP struct
mutable struct EPABC{X,M}
    data ::X
    model::M
    sites::Vector{M}
    M    ::Int
    λ    ::Float64  # for damped update...
end

function EPABC(data, model::T; M=10000, λ=1.) where T
    sites = Vector{T}(undef, length(data))
    EPABC(data, model, sites, M, λ)
end

# 1. We need a method to form the cavity density
# it should not be a problem that this modifies the full approximation
function getcavity(full, site)
    q = deepcopy(full.q)
    for (clade, η) in site.q.cmap
        q.cmap[clade] -= η
    end
    Ψ = cavity(full.Ψ, site.Ψ)
    return MSCModel(Ψ, q, full.m1, full.m2)
end

# 2. We need a method to simulate from the cavity, basically a method
# to simulate MSC models from the MSCModel or whatever we call it
function randsptree(model::MSCModel)
    S = randtree(model.Ψ)  # a random tree topology
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

function normalize!(bmp)
    for (k,d) in bmp.smap
        Z = sum(values(d))
        for (s,v) in d
            d[s] /= Z
        end
    end
end

# 3. We need a method to update the full approximation by moment
# matching. We need something like the BranchModel storing for each
# clade in the accepted simulations the count for that clade, and its
# mean and squared parameter value
function updated_model(accepted_trees, model, λ)
    Ψ = newbmp(accepted_trees, model, λ)
    q = newbranches(Ψ, accepted_trees, model, λ)
    MSCModel(Ψ, q, model.m1, model.m2)
    #MSCModel(Ψ, model.q, model.m1, model.m2)
end

function newbmp(accepted_trees, model, λ)
    Ψ = fitbmp(accepted_trees, model.m2)
    # damping **assuming full support is represented**
    for (k,d) in model.Ψ.smap
        for (s, p) in d
            if !haskey(Ψ.smap, k)
                Ψ.smap[k] = Dict(s=>(1-λ)*p)
            elseif !haskey(Ψ.smap[k], s)
                Ψ.smap[k][s] = (1-λ)*p
            else
                Ψ.smap[k][s] *= λ
                Ψ.smap[k][s] += (1-λ)*p
            end
        end
    end
    normalize!(Ψ)
    return Ψ
end

function newbranches(Ψ, accepted_trees, model, λ)
    d = Dict(clade=>zeros(3) for clade in keys(Ψ.smap))
    for tree in accepted_trees
        _record_branchparams!(d, tree, model.m2)
    end
    q′ = BranchModel(update_q(d, model.q, λ), model.q.prior)
    return q′
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

function update_q(q′, q, λ)
    d = Dict(clade => mom2nat(x...) for (clade, x) in q′ if !(all(x .== 0.)))
    for (clade, x) in d
        y = haskey(q.cmap, clade) ? q.cmap[clade] : q.prior
        d[clade] = ((1-λ) .* y) .+ (λ .* x)
    end
    return d
end

mom2nat(n, xs, xsqs) = mom2nat(xs/n, xsqs/n)
mom2nat(μ, V) = [μ/V, -1.0/(2V)]

new_site(new_full, cavity) = getcavity(new_full, cavity)

function ep_iteration!(alg, i, minacc=1)
    @unpack data, model, sites, M = alg
    x = data[i]
    cavity = isassigned(sites, i) ? getcavity(model, sites[i]) : model
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
    nacc < minacc && return alg.model
    model_ = updated_model(last.(accepted), model, alg.λ)
    #alg.sites[i] = new_site(model_, cavity) 
    alg.model = model_
end





