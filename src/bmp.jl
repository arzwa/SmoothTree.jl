# We have use for an abstract type here
abstract type AbstractBMP{T,V} end

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

A BMP model in moment parameter space.
"""
struct MomBMP{T,V} <: AbstractBMP{T,V}
    smap::Dict{T,SparseSplits{T,V}}
    root::T
end

# accessors
Base.show(io::IO, m::M) where M<:AbstractBMP = write(io, "$M(o=$(m.root))")
Base.haskey(m::AbstractBMP, γ) = haskey(m.smap, γ)
Base.haskey(m::AbstractBMP, γ, δ) = haskey(m, γ) && haskey(m.smap[γ], δ)
Base.getindex(m::AbstractBMP, γ) = m.smap[γ]
Base.getindex(m::AbstractBMP, γ, δ) = m.smap[γ][δ]

# 'empty' BMP (uniform on splits, this does *not* correspond to the
# zero BMP...)
const SplitDict{T} = Dict{T,SparseSplits{T,Float64}}
NatBMP(root::T) where T<:Integer = NatBMP(SplitDict{T}(), root)
MomBMP(root::T) where T<:Integer = MomBMP(SplitDict{T}(), root)

# natural parameter -> moment parameter
MomBMP(x::NatBMP) = MomBMP(Dict(k=>nat2mom(x) for (k,x) in x.smap), x.root)
NatBMP(x::MomBMP) = NatBMP(Dict(k=>mom2nat(x) for (k,x) in x.smap), x.root)

# BMP with fixed outgroup, note we do not set the P to exactly 1,
# because that leads to ill-defined NatBMPs
NatBMP(root, rootsplit) = NatBMP(MomBMP(root, rootsplit))
function MomBMP(root::T, rootsplit::T, ϵ=1e-16) where T
    split = min(rootsplit, root - rootsplit)
    d = Dict(split => 1. - ϵ)
    ρ = refsplit(root)
    n = _ns(cladesize(root))
    η0 = ϵ/(n-1)
    roots = SparseSplits(d, n, 1, η0, ρ)
    MomBMP(Dict(root=>roots), root)
end

# get the mean BMP implied by a Dirichlet-BMP
MomBMP(x::CCD) = MomBMP(NatBMP(x))
function NatBMP(x::CCD)
    smap = Dict(γ=>SparseSplits(γ, d, x.α) for (γ, d) in x.smap)
    NatBMP(smap, x.root)
end

# linear operations
function Base.:+(x::NatBMP{T,V}, y::NatBMP{T,V}) where {T,V}
    d = Dict(γ=>v for (γ, v) in x.smap)
    for (γ, v) in y.smap
        d[γ] = haskey(d, γ) ? d[γ] + v : v
    end
    NatBMP(d, x.root) 
end

function Base.:-(x::NatBMP{T,V}, y::NatBMP{T,V}) where {T,V}
    d = Dict(γ=>v for (γ, v) in x.smap)
    for (γ, v) in y.smap
        d[γ] = haskey(d, γ) ? d[γ] - v : -1.0*v
    end
    NatBMP(d, x.root) 
end

function Base.:*(x::NatBMP{T,V}, a::V) where {T,V}
    NatBMP(Dict(γ=>v*a for (γ,v) in x.smap), x.root)
end

# randtree for moment parameters
randtree(model::MomBMP) = _randwalk(Node(model.root), model)

# randtree for natural parameter SparseSplits  # FIXME? 
# I guess no need to fix this, but we should take this into account in
# the EP algorithm by only converting the cavity once to moment space
# and simulating species trees from that MomBMP...
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

# XXX for moment parameter!
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

"""
    prune

Prune a sparsely represented BMP object by setting all represented
splits with split probabilities indistinguishable from the probability
of an unrepresented split to the latter (thereby removing the split
from the set of explicitly represented splits).
"""
function prune(x::M; atol=1e-9) where {T,V,M<:AbstractBMP{T,V}}
    newd = Dict{T,SparseSplits{T,V}}()
    for (γ, x) in x.smap
        newd[γ] = prune(x, atol)
    end
    return M(newd, x.root)
end

