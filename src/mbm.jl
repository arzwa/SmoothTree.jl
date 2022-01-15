# We have use for an abstract type here
abstract type AbstractMBM{T,V} end

# this is no more than a container of SparseSplits distributions
"""
    NatMBM

A MBM model in natural parameter space.
"""
struct NatMBM{T,V} <: AbstractMBM{T,V}
    β::V
    smap::Dict{T,SparseSplits{T,V}}
    root::T
end

"""
    MomMBM

A MBM model in moment parameter space.
"""
struct MomMBM{T,V} <: AbstractMBM{T,V}
    β::V
    smap::Dict{T,SparseSplits{T,V}}
    root::T
end

# accessors
Base.show(io::IO, m::M) where M<:AbstractMBM = write(io, "$M(o=$(m.root))")
Base.haskey(m::AbstractMBM, γ) = haskey(m.smap, γ)
Base.haskey(m::AbstractMBM, γ, δ) = haskey(m, γ) && haskey(m.smap[γ], δ)
Base.getindex(m::AbstractMBM, γ) = m.smap[γ]
Base.getindex(m::AbstractMBM, γ, δ) = m.smap[γ][δ]

# 'empty' MBM (uniform on splits, this does *not* correspond to the
# zero MBM...)
const SplitDict{T} = Dict{T,SparseSplits{T,Float64}}
NatMBM(root::T, β) where T<:Integer = NatMBM(β, SplitDict{T}(), root)
MomMBM(root::T, β) where T<:Integer = MomMBM(β, SplitDict{T}(), root)

# natural parameter -> moment parameter
MomMBM(x::NatMBM) = MomMBM(x.β, Dict(k=>nat2mom(x) for (k,x) in x.smap), x.root)
NatMBM(x::MomMBM) = NatMBM(x.β, Dict(k=>mom2nat(x) for (k,x) in x.smap), x.root)

# MBM with fixed outgroup, note we do not set the P to exactly 1,
# because that leads to ill-defined NatMBMs
#NatMBM(root, rootsplit) = NatMBM(MomMBM(root, rootsplit))
#function MomMBM(root::T, rootsplit::T, ϵ=1e-16) where T
#    split = min(rootsplit, root - rootsplit)
#    d = Dict(split => 1. - ϵ)
#    ρ = refsplit(root)
#    n = _ns(cladesize(root))
#    η0 = ϵ/(n-1)
#    roots = SparseSplits(d, n, 1, η0, ρ)
#    MomMBM(Dict(root=>roots), root)
#end

# get the mean MBM implied by a Dirichlet-MBM
MomMBM(x::CCD, args...) = MomMBM(NatMBM(x, args...))
function NatMBM(x::CCD, β, α)
    smap = Dict(γ=>SparseSplits(γ, d, β, α) for (γ, d) in x.smap)
    NatMBM(β, smap, x.root)
end

# linear operations
function Base.:+(x::NatMBM{T,V}, y::NatMBM{T,V}) where {T,V}
    d = Dict(γ=>v for (γ, v) in x.smap)
    for (γ, v) in y.smap
        d[γ] = haskey(d, γ) ? d[γ] + v : v
    end
    NatMBM(d, x.root) 
end

function Base.:-(x::NatMBM{T,V}, y::NatMBM{T,V}) where {T,V}
    d = Dict(γ=>v for (γ, v) in x.smap)
    for (γ, v) in y.smap
        d[γ] = haskey(d, γ) ? d[γ] - v : -1.0*v
    end
    NatMBM(d, x.root) 
end

function Base.:*(x::NatMBM{T,V}, a::V) where {T,V}
    NatMBM(Dict(γ=>v*a for (γ,v) in x.smap), x.root)
end

# randtree for moment parameters
randtree(model::MomMBM) = _randwalk(Node(model.root), model)

# randtree for natural parameter SparseSplits  # FIXME? 
# I guess no need to fix this, but we should take this into account in
# the EP algorithm by only converting the cavity once to moment space
# and simulating species trees from that MomMBM...
randtree(model::NatMBM) = randtree(MomMBM(model))
randtree(model::NatMBM, n) = randtree(MomMBM(model), n)

# recursion for randtree
function _randwalk(node, model::MomMBM)
    clade = id(node)
    isleafclade(clade) && return
    splt = randsplit(model, clade)
    n1 = Node(splt, node)
    n2 = Node(clade - splt, node)
    _randwalk(n1, model)
    _randwalk(n2, model)
    return node
end

function randsplit(m::MomMBM, γ)
    # a cherry clade has NaN entries in SparseSplits
    haskey(m, γ) && !ischerry(γ) ? randsplit(m[γ]) : randsplit(γ, m.β)
end

#"""
#    prune
#
#Prune a sparsely represented MBM object by setting all represented
#splits with split probabilities indistinguishable from the probability
#of an unrepresented split to the latter (thereby removing the split
#from the set of explicitly represented splits).
#"""
#function prune(x::M, atol) where {T,V,M<:AbstractMBM{T,V}}
#    newd = Dict{T,SparseSplits{T,V}}()
#    clades = Set(x.root)
#    # first we prune all splits with negligible probability
#    for (γ, x) in x.smap
#        newd[γ] = prune(x, atol)
#        union!(clades, keys(newd[γ].splits))
#    end
#    # then we prune clades which feature explicitly in none of the
#    # split distributions
#    toprune = setdiff(keys(newd), clades)
#    for γ in toprune
#        delete!(newd, γ)
#    end
#    return M(newd, x.root)
#end
#
#function prune!(x::M, atol) where {T,V,M<:AbstractMBM{T,V}}
#    clades = Set(x.root)
#    for (γ, x) in x.smap
#        prune!(x, atol)
#        union!(clades, keys(x.splits))
#    end
#    toprune = setdiff(keys(x.smap), clades)
#    for γ in toprune
#        delete!(x.smap, γ)
#    end
#end


