
abstract type BMP{T,V} end

# this is how the natural parameterization of a Categorical should be
# implemented (need a k-1 parameter vector for k categories)
struct CatNat{T}
    η::Vector{T}
end
# note that in this parameterization, we cannot have pk=0 or pk=1 for
# any k...

lognormalizer(d::CatNat) = 1. + exp.(d)

function Distributions.Categorical(d::CatNat)
    ps = exp.(d.η)
    Z = 1 + sum(ps)
    Distributions.Categorical([ps ./ Z ; [1/Z]]) 
end

# Naively, we might think we just put the unrepresented splits in a
# single category and that's that. That's quite alright, but we need
# to be a little more subtle when constructing the cavity for
# instance, when some split is explicitly represented in one but not
# the other Categorical distribution.
# The n-k  unrepresented splits have the same probability, that is the
# only additional information we'd need right?
# Turns out it will be easier not to lump together the unrepresented
# splits as a single additional category, as that makes dealing with
# the cavity tricky.

refsplit(γ::T) where T = T(2^(ndigits(γ, base=2) - 1))

# used for both parameterizations
struct SparseSplits{T,V}
    splits::Dict{T,V}
    n  ::Int
    k  ::Int
    η0 ::V
    ref::T
end

# this is no more than a container of SparseSplits distributions
struct NatBMP{T,V}
    smap::Dict{T,SparseSplits{T,V}}
    root::T
end

struct MomBMP{T,V}
    smap::Dict{T,SparseSplits{T,V}}
    root::T
end

# get the mean BMP implied by a Dirichlet-BMP
function NatBMP(x::CCD)
    smap = Dict(γ=>SparseSplits(γ, d, x.α) for (γ, d) in x.smap)
    NatBMP(smap, x.root)
end

# Get a natural parameter SparseSplits object from a split
# distribution
function SparseSplits(γ, d::Dict{T,V}) where {T,V}
    ρ = refsplit(γ)
    n = _ns(cladesize(γ))
    k = length(d)
    p = sum(values(d))
    p0 = log((1 - p)/(n - k))  # probability of unrepresented split
    pρ = haskey(d, ρ) ? log(d[ρ]) : p0
    η0 = p0 - pρ
    dd = Dict(δ => log(p) - pρ for (δ, p) in d)
    SparseSplits(dd, n, k, η0, ρ)
end

# from a Dirichlet BMP
# XXX double check
function SparseSplits(γ, d::Dict{T,V}, α) where {T,V}
    ρ = refsplit(γ)
    n = _ns(cladesize(γ))
    k = length(d)
    N = sum(values(d))
    Z = n*α + N
    pr = (k*α + N)/Z  # represented pr mass
    p0 = log((1 - pr)/(n - k))  # probability of unrepresented split
    pρ = haskey(d, ρ) ? log(α + d[ρ]) - log(Z) : p0
    η0 = p0 - pρ
    dd = Dict(δ => log(α + p) - log(Z) - pρ for (δ, p) in d)
    SparseSplits(dd, n, k, η0, ρ)
end

# natural parameter -> moment parameter
MomBMP(x::NatBMP) = MomBMP(Dict(k=>nat2mom(x) for (k,x) in x.smap), x.root)
NatBMP(x::MomBMP) = NatBMP(Dict(k=>mom2nat(x) for (k,x) in x.smap), x.root)

function nat2mom(x::SparseSplits)
    ρ = x.ref
    d = Dict(k=>exp(v) for (k,v) in x.splits)
    S = sum(values(d))
    Z = if haskey(x.splits, ρ)
        1 + S + (x.n - x.k) * exp(x.η0) - exp(x.splits[ρ])
    else
        1 + S + (x.n - x.k - 1) * exp(x.η0)
    end
    for (k, v) in d
        d[k] /= Z
    end
    return SparseSplits(d, x.n, x.k, exp(x.η0)/Z, ρ)
end

function mom2nat(x::SparseSplits)
    ρ = x.ref
    pρ = haskey(x.splits, ρ) ? log(x.splits[ρ]) : log(x.η0)
    η0 = log(x.η0) - pρ
    d = Dict(k=>log(p) - pρ for (k,p) in x.splits)
    return SparseSplits(d, x.n, x.k, η0, ρ)
end

# randtree for moment parameters
randtree(model::MomBMP) = _randwalk(Node(model.root), model)

# randtree for natural parameter SparseSplits  # FIXME?
randtree(model::NatBMP) = randtree(MomBMP(model))
randtree(model::NatBMP, n) = randtree(MomBMP(model), n)

# recursion for randtree
function _randwalk(node, model::MomBMP)
    clade = id(node)
    isleafclade(clade) && return
    splt = randsplit(model, clade)
    n1 = Node(splt, node)
    n2 = Node(clade - splt, node)
    _randwalk(n1, model)
    _randwalk(n2, model)
    return node
end

function randsplit(model::MomBMP, γ)
    if haskey(model.smap, γ) && !ischerry(γ) 
        randsplit(model.smap[γ], γ) 
    else 
        randsplit(γ)
    end
end

# for moment parameter...
function randsplit(x::SparseSplits, γ)
    splitps = collect(x.splits)
    weights = last.(splitps)
    splits  = first.(splitps)
    if rand() < sum(weights)
        i = sample(1:length(weights), Weights(weights))
        return splits[i]
    else  # rejection sampler...
        δ = randsplit(γ)
        #while δ ∈ splits
        #    δ = randsplit(γ)
        #end
        return δ
    end
end

# construct a cavity distribution
# NOTE: a site should never have more clades than the global
# approximation
# TODO: perhaps worthwhile to have also cavity! (i.e. modify inplace)
function cavity(full::NatBMP{T,V}, site::NatBMP{T,V}) where {T,V}
    newd = Dict{T,SparseSplits{T,V}}()
    for (clade, d) in full.smap  # site never has more clades than global
        newd[clade] = haskey(site.smap, clade) ? 
            _cavity(d, site.smap[clade]) : d
    end
    return NatBMP(newd, full.root)
end

function _cavity(full::SparseSplits{T,V}, site) where {T,V}
    d = Dict{T,V}()
    for (δ, η) in full.splits
        ηi = haskey(site.splits, δ) ? site.splits[δ] : site.η0
        d[δ] = η - ηi
    end
    η0 = full.η0 - site.η0
    SparseSplits(d, full.n, full.k, η0, full.ref)
end

# TODO: should implement several algebraic rules for SparseSplits and
# make it more elegant and bug proof
function convexcombination(x::NatBMP{T,V}, y::NatBMP{T,V}, λ) where {T,V}
    newd = Dict{T,SparseSplits{T,V}}()
    clades = union(keys(x.smap), keys(y.smap))
    for clade in clades
        newd[clade] = if haskey(x.smap, clade) && haskey(y.smap, clade)
            _convexcombination(x.smap[clade], y.smap[clade], λ)
        elseif haskey(x.smap, clade)
            _convexcombination(x.smap[clade], λ)
        else
            _convexcombination(y.smap[clade], λ)
        end
    end    
    return NatBMP(newd, x.root)
end

function _convexcombination(x::SparseSplits{T,V}, y, λ) where {T,V}
    d = Dict{T,V}()
    splits = union(keys(x.splits), keys(y.splits))
    for δ in splits
        ηx = haskey(x.splits, δ) ? x.splits[δ] : x.η0
        ηy = haskey(y.splits, δ) ? y.splits[δ] : y.η0
        d[δ] = λ*ηx + (1-λ)*ηy
    end
    η0 = λ*x.η0 - (1-λ)*y.η0
    SparseSplits(d, x.n, length(splits), η0, x.ref) 
end

function _convexcombination(x::SparseSplits{T,V}, λ) where {T,V}
    d = Dict(k=>λ*η for (k,η) in x.splits)
    SparseSplits(d, x.n, x.k, λ*x.η0, x.ref) 
end

