abstract type AbstractBMP{T,V} end

# we pick a fixed *reference split* which serves as the supremum in the
# order of splits. This is defined to be the leaf clade with largest
# clade index
refsplit(γ::T) where T = T(2^(ndigits(γ, base=2) - 1))

# used for both natural and moment parameterizations
"""
    SparseSplits

Sparse representation of a split distribution, where all
non-explicitly stored splits are assumed to have equal probability.
"""
struct SparseSplits{T,V}
    splits::Dict{T,V}  # explicitly represented splits
    n  ::Int           # total number of splits
    k  ::Int           # number of explicitly represented splits
    η0 ::V             # parameter for unrepresented splits
    ref::T             # reference split
end

# accessors
Base.haskey(m::SparseSplits, δ) = haskey(m.splits, δ)
Base.getindex(m::SparseSplits, δ) = haskey(m.splits, δ) ? m.splits[δ] : m.η0

# this is no more than a container of SparseSplits distributions
"""
    NatBMP

A BMP model in natural parameter space.
"""
struct NatBMP{T,V} <: AbstractBMP{T,V}
    smap::Dict{T,SparseSplits{T,V}}
    root::T
end

"""
    MomBMP

A BMP model in moment parameter space
"""
struct MomBMP{T,V} <: AbstractBMP{T,V}
    smap::Dict{T,SparseSplits{T,V}}
    root::T
end

# accessors
Base.haskey(m::AbstractBMP, γ) = haskey(m.smap, γ)
Base.haskey(m::AbstractBMP, γ, δ) = haskey(m, γ) && haskey(m.smap[γ], δ)
Base.getindex(m::AbstractBMP, γ) = m.smap[γ]
Base.getindex(m::AbstractBMP, γ, δ) = m.smap[γ][δ]

# get the mean BMP implied by a Dirichlet-BMP
function NatBMP(x::CCD)
    smap = Dict(γ=>SparseSplits(γ, d, x.α) for (γ, d) in x.smap)
    NatBMP(smap, x.root)
end

"""
Get a natural parameter SparseSplits object from a split distribution
`d`, assuming the full support is represented in `d`.
"""
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

"""
Get a natural parameter SparseSplits object from split counts `d`,
assuming the Dirichlet-BMP model with Dirichlet parameter `α`
"""
function SparseSplits(γ, d::Dict{T,V}, α) where {T,V}
    ρ = refsplit(γ)
    n = _ns(cladesize(γ))  # total number of splits
    k = length(d)          # number of represented splits
    pρ = haskey(d, ρ) ? log(α + d[ρ]) : log(α)  # unnormalized pr of split ρ
    η0 = log(α) - pρ 
    dd = Dict(δ => log(α + c) - pρ for (δ, c) in d)
    SparseSplits(dd, n, k, η0, ρ)
end

# natural parameter -> moment parameter
MomBMP(x::NatBMP) = MomBMP(Dict(k=>nat2mom(x) for (k,x) in x.smap), x.root)
NatBMP(x::MomBMP) = NatBMP(Dict(k=>mom2nat(x) for (k,x) in x.smap), x.root)

function nat2mom(x::SparseSplits)
    ρ = x.ref
    d = Dict(k=>exp(v) for (k,v) in x.splits)
    S = sum(values(d))
    Z = S + (x.n - x.k) * exp(x.η0) 
    for (k, v) in d
        d[k] /= Z
    end
    return SparseSplits(d, x.n, x.k, exp(x.η0)/Z, ρ)
end

function mom2nat(x::SparseSplits)
    ρ = x.ref
    pρ = log(x[ρ])
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

function randsplit(m::MomBMP, γ)
    # a cherry clade has NaN entries in SparseSplits
    haskey(m, γ) && !ischerry(γ) ? randsplit(m[γ], γ) : randsplit(γ)
end

# XXX for moment parameter
function randsplit(x::SparseSplits, γ)
    splitps = collect(x.splits)
    weights = last.(splitps)
    splits  = first.(splitps)
    pr = weights .- x.η0  
    # pr is the P mass of represented splits after taking out uniform
    # note that we simulate by splitting the categorical distribution
    # in a uniform and non-uniform component with restricted support
    if rand() < sum(pr)
        i = sample(1:length(pr), Weights(pr))
        return splits[i]
    else 
        return randsplit(γ)
    end
end

# construct a cavity distribution
# NOTE: a site should never have more clades than the global
# approximation
# TODO: perhaps worthwhile to have also cavity! (i.e. modify inplace)
function cavity(full::NatBMP{T,V}, site::NatBMP{T,V}) where {T,V}
    newd = Dict{T,SparseSplits{T,V}}()
    for (clade, d) in full.smap  # site never has more clades than global
        newd[clade] = haskey(site, clade) ? _cavity(d, site[clade]) : d
    end
    return NatBMP(newd, full.root)
end

function _cavity(full::SparseSplits{T,V}, site) where {T,V}
    d = Dict{T,V}()
    for (δ, η) in full.splits
        d[δ] = η - site[δ]
    end
    η0 = full.η0 - site.η0
    SparseSplits(d, full.n, full.k, η0, full.ref)
end

function convexcombination(x::NatBMP{T,V}, y::NatBMP{T,V}, λ) where {T,V}
    newd = Dict{T,SparseSplits{T,V}}()
    clades = union(keys(x.smap), keys(y.smap))
    for clade in clades
        newd[clade] = if haskey(x, clade) && haskey(y, clade)
            _convexcombination(x[clade], y[clade], λ)
        elseif haskey(x, clade)
            _convexcombination(x[clade], λ)
        else
            _convexcombination(y[clade], 1-λ)
        end
    end    
    return NatBMP(newd, x.root)
end

function _convexcombination(x::SparseSplits{T,V}, y, λ) where {T,V}
    d = Dict{T,V}()
    splits = union(keys(x.splits), keys(y.splits))
    for δ in splits
        d[δ] = λ*x[δ] + (1-λ)*y[δ]
    end
    η0 = λ*x.η0 - (1-λ)*y.η0
    SparseSplits(d, x.n, length(splits), η0, x.ref) 
end

function _convexcombination(x::SparseSplits{T,V}, λ) where {T,V}
    d = Dict(k=>λ*η for (k,η) in x.splits)
    SparseSplits(d, x.n, x.k, λ*x.η0, x.ref) 
end

