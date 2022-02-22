# Arthur Zwaenepoel 2021

# Note: clades are simply represented by integers, whose binary expansion is a
# bitstring recording presence/absence of leaf clades.  In other words the
# clade with id 13 is clade 1101, representing the clade containing leaf 4,
# leaf 3 and leaf 1. This restricts us here to trees with at most 62 leaves.

# Note: trees are currently represented as rooted

"""
    CCD(trees::Vector; [lmap=taxon_map])
    CCD(trees::AbstractDict; [lmap=taxon_map])

A conditional clade distribution (CCD) object. 

Input data can be either a vector of trees or a countmap of trees (see
`StatsBase.countmap`).

# Examples
```jldoctest
julia> ccd = CCD([nw"((A,B),C);", nw"((B,C),A);", nw"((A,B),C);"])
CCD{UInt16}(n=3, Γ=7)
```
"""
struct CCD{T,V}
    cmap::Dict{T,V}          # clade counts
    smap::Dict{T,Dict{T,V}}  # split counts
    root::T
end

# some Base methods
Base.haskey(x::CCD, γ) = haskey(x.cmap, γ)
Base.haskey(x::CCD, γ, δ) = haskey(x.smap, γ) && haskey(x.smap[γ], δ)
Base.getindex(x::CCD, γ) = haskey(x, γ) ? x.cmap[γ] : 0
Base.getindex(x::CCD, γ, δ) = haskey(x, γ, δ) ? x.smap[γ][δ] : 0
Base.show(io::IO, X::CCD) = write(io, "CCD(Γ=$(X.root))")
showclades(X::CCD) = sort([(k, bitstring(k), v) for (k,v) in X.cmap])

# check if some bitclade represents a leaf
isleafclade(clade) = count_ones(clade) == 1

# check if cherry clade
ischerry(clade) = count_ones(clade) == 2

# get the number of leaves in a clade
cladesize(clade) = count_ones(clade)

# get the root clade for n leaves
rootclade(n, T=UInt64) = T(2^n - 1) 

# conditional clade probability
ccp(ccd::CCD, γ, δ) = ccd[γ,δ] / ccd[γ]
logccp(ccd, γ, δ) = log(ccd[γ,δ]) - log(ccd[γ])


# Constructors
# ------------
# from a countmap
CCD(trees::AbstractDict, lmap, ::Type{T}; rooted=true) where T = 
    CCD(collect(trees), lmap, T, rooted=rooted)

# For a single tree
CCD(tree::Node, args...; kwargs...) = CCD([tree], args...; kwargs...)

# from a vector of (pairs of) trees
function CCD(trees, lmap, ::Type{T}=Float64; rooted=true) where T
    ccd = initccd(lmap, T)
    for tree in trees
        rooted ? addtree!(ccd, lmap, tree) : addunrooted!(ccd, lmap, tree)
    end
    return ccd
end

# initialize a ccd
function initccd(lmap::BiMap{T,V}, ::Type{W}) where {T,V,W}
    cmap = Dict{T,W}(γ=>zero(W) for (γ,_) in lmap)
    smap = Dict{T,Dict{T,W}}()
    root = T(sum(keys(lmap)))
    cmap[root] = zero(W)  
    smap[root] = Dict{T,W}()
    ccd = CCD(cmap, smap, root)
end

# add a tree/number of identical trees to the CCD
addtree!(ccd::CCD, lmap, tpair) = addtree!(ccd, lmap, tpair[1], tpair[2])  
function addtree!(ccd::CCD, lmap, tree::Node, w=1)
    @unpack cmap, smap = ccd
    @assert NewickTree.isbifurcating(tree)
    function walk(n)
        if isleaf(n)
            leaf = lmap[name(n)] 
            cmap[leaf] += w
            return leaf
        else
            left = walk(n[1])
            rght = walk(n[2])
            clade = left + rght
            x = min(left, rght)
            _update!(cmap, smap, clade, x)
            cmap[clade] += w
            smap[clade][x] += w
            return clade
        end
    end
    walk(tree)
end

# NOTE, we assume the unrooted tree is represented as a rooted one!
# XXX much too slow for big trees...
addunrooted!(ccd::CCD, lmap, tpair) = addunrooted!(ccd, lmap, tpair[1], tpair[2])  
function addunrooted!(ccd::CCD, lmap, tree::Node, w=1)
    @assert NewickTree.isbifurcating(tree)
    o = prewalk(tree)
    m = length(o) - 2
    for n in o
        parent(n) == tree && continue
        t = set_outgroup(n)
        addtree!(ccd, lmap, t, w/m)
    end
end

# For branches... actually very similar for splits... assumes rooted!
function CCD(bs::Vector{Branches{T}}, ws::Vector{W}) where {T,W}
    ccd = initccd(bs[1], W)
    for (x, w) in zip(bs, ws)
        addtree!(ccd, x, w)
    end
    return ccd
end

function initccd(b::Branches{T}, ::Type{W}) where {T,W}
    root = b[1][1]  # XXX should be in preorder!
    cmap = Dict{T,W}()
    smap = Dict{T,Dict{T,W}}()
    ccd = CCD(cmap, smap, root)
end

function addtree!(ccd::CCD, x::Branches, w)
    for i=length(x):-2:1
        γ, δ, _ = x[i]
        δ = min(δ, γ-δ)
        _update!(ccd.cmap, ccd.smap, γ, δ)
        ccd.cmap[γ] += w
        ccd.smap[γ][δ] += w
    end
end

# should we use defaultdict instead?
function _update!(m1, m2, y, x)
    !haskey(m1, y) && (m1[y] = 0)
    if !haskey(m2, y) 
        m2[y] = Dict(x=>0)
    elseif !haskey(m2[y], x)
        m2[y][x] = 0
    end
end


# Sampling methods
# ----------------
# do n `randsplits` simulations
randsplits(model, n) = map(_->randsplits(model), 1:n)

# draw a tree from the ccd, simulates a tree as a set of splits
randsplits(ccd::CCD{T}) where T = _randwalk(Tuple{T,T}[], ccd.root, ccd)

function _randwalk(splits, clade, ccd)
    isleafclade(clade) && return splits
    csplits = collect(ccd.smap[clade])
    splt = sample(1:length(csplits), Weights(last.(csplits)))
    left = first(csplits[splt])
    rght = clade - left
    push!(splits, (clade, left))
    splits = _randwalk(splits, left, ccd)
    splits = _randwalk(splits, rght, ccd)
    return splits
end

# draw a random tree from the CCD
function randtree(ccd::CCD, lmap)
    splits = randsplits(ccd)
    gettree(splits, lmap)
end

randtree(model::CCD, m, n) = map(_->randtree(model, m), 1:n)


# Compute probabilities
# ---------------------
# compute the probability of a set of splits
function logpdf(ccd::CCD, splits::Vector{T}) where T<:Tuple
    ℓ = 0.
    for (γ, δ) in splits
        ℓ += logccp(ccd, γ, δ)
    end
    return isnan(ℓ) ? -Inf : ℓ
end
# XXX: Ideally, we'd check in some way whether the leaf set of the ccd
# and check if splits actually correspond to splits in the ccd, if not
# return -Inf, but that induces an overhead I guess...

# compute the probability mass of a single tree under the CCD
function logpdf(ccd::CCD, lmap, tree::Node)
    ℓ, _ = _lwalk(tree, ccd, lmap, 0.)
    return isnan(ℓ) ? -Inf : ℓ
end

function _lwalk(n::Node, ccd, lmap, ℓ)
    isleaf(n) && return ℓ, lmap[name(n)]
    ℓ, left = _lwalk(n[1], ccd, lmap, ℓ) 
    ℓ, rght = _lwalk(n[2], ccd, lmap, ℓ)
    δ = left < rght ? left : rght
    γ = left + rght
    ℓ += logccp(ccd, γ, δ) 
    return ℓ, γ
end

# for a vector of trees
logpdf(ccd::CCD, trees::AbstractVector) = mapreduce(t->logpdf(ccd, t), +, trees)

# for a countmap
function logpdf(ccd::CCD, trees::Dict)
    l = 0.
    for (tree, count) in trees
        l += count*logpdf(ccd, tree)
    end
    return l
end


