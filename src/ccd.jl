# Arthur Zwaenepoel 2021

# Rooted/unrooted? If all input trees are (arbitrarily) rooted at some
# taxon, when α=0, we'll have that all trees from the CCD are rooted
# in the same arbitrary way, and they can be interpreted as draws from
# an unrooted tree distribution. Now when α>0, this is no longer the
# case, and if we don't explicitly distinguish the rooted case from
# the unrooted we'll have different distributions. In the unrooted
# case, represented by an arbitrary rooting, we should have that the
# root split is certain, despite α>1 (this sounds a bit confusing).

# Note: clades are simply represented by integers, whose binary
# expansion is a bitstring recording presence/absence of leaf clades.
# In other words the clade with id 13 is clade 1101, representing the
# clade containing of leaf 4, leaf 3 and leaf 1.

# XXX reconsider the taxon map stuff
# We might do away with it altogether, requiring appropriately labeled
# trees as input?
#
# XXX refactor to keep CCD as in Larget, separating it from the prior
# model, i.e. CCD is just a data structure for observed splits
#
# XXX smap needlessly stores cherries

# aliases
const DefaultNode{T} = Node{T,NewickData{Float64,String}}
const Splits{T} = Vector{Tuple{T,T}}

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
    lmap::BiMap{T,String}
    cmap::Dict{T,V}          # clade counts
    smap::Dict{T,Dict{T,V}}  # split counts
    root::T
end

# some Base methods
Base.haskey(x::CCD, γ) = haskey(x.cmap, γ)
Base.haskey(x::CCD, γ, δ) = haskey(x.smap, γ) && haskey(x.smap[γ], δ)
Base.getindex(x::CCD, γ) = haskey(x, γ) ? x.cmap[γ] : 0
Base.getindex(x::CCD, γ, δ) = haskey(x, γ, δ) ? x.smap[γ][δ] : 0
Base.show(io::IO, X::CCD) = write(io, "CCD(n=$(length(X.lmap)), Γ=$(X.root))")
showclades(X::CCD) = sort([(k, bitstring(k), v) for (k,v) in X.cmap])

# leaf names
NewickTree.getleaves(x::CCD) = collect(keys(x.lmap))

# check if some bitclade represents a leaf
isleafclade(clade) = count_ones(clade) == 1

# check if cherry clade
ischerry(clade) = count_ones(clade) == 2

# get the number of leaves in a clade
cladesize(clade) = count_ones(clade)

# get the root clade for n leaves
rootclade(n, T=UInt16) = T(2^n - 1) 

# conditional clade probability
ccp(ccd::CCD, γ, δ) = ccd[γ,δ] / ccd[γ]
logccp(ccd, γ, δ) = log(ccd[γ,δ]) - log(ccd[γ])

# uniform weights for a given vector of elements 
uweights(xs) = fill(1/length(xs), length(xs))

# Constructors
# from a countmap
CCD(trees::AbstractDict, args...) = CCD(collect(trees), args...)

# For a single tree
CCD(tree::Node, args...) = CCD([tree], args...)

# from a vector of (pairs of) trees
function CCD(trees, lmap)
    ccd = initccd(lmap)
    for tree in trees
        addtree!(ccd, tree)
    end
    return ccd
end

# initialize a ccd
function initccd(lmap::BiMap{T,V}) where {T,V}
    cmap = Dict{T,Int64}(γ=>0 for (γ,_) in lmap)
    smap = Dict{T,Dict{T,Int64}}()
    root = T(sum(keys(lmap)))
    cmap[root] = 0  
    smap[root] = Dict{T,Int64}()
    ccd = CCD(lmap, cmap, smap, root)
end

# add a tree/number of identical trees to the CCD
addtree!(ccd::CCD, tpair) = addtree!(ccd, tpair[1], tpair[2])  
function addtree!(ccd::CCD, tree::Node, w=1)
    @unpack lmap, cmap, smap = ccd
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

# should we use defaultdict instead?
function _update!(m1, m2, y, x)
    !haskey(m1, y) && (m1[y] = 0)
    if !haskey(m2, y) 
        m2[y] = Dict(x=>0)
    elseif !haskey(m2[y], x)
        m2[y][x] = 0
    end
end

# get the modal tree? this uses a greedy algorithm, not sure if
# guaranteed to give the mode?
function modetree(ccd::CCD{T}) where T
    l, splits = _modetree(0., Tuple{T,T}[], ccd, ccd.root)
end

function _modetree(l, splits, ccd, γ)
    isleafclade(γ) && return l, splits
    xs = collect(ccd.smap[γ])
    i = argmax(last.(xs))
    δ = xs[i][1]
    l += _splitp(ccd, γ, δ)
    push!(splits, (γ, δ))
    l, splits = _modetree(l, splits, ccd, δ)
    l, splits = _modetree(l, splits, ccd, γ-δ)
    return l, splits
end

# draw a tree from the ccd, simulates a tree as a set of splits
function randsplits(ccd::CCD{T}) where T
    _randwalk1(Tuple{T,T}[], ccd.root, ccd)
end

# do n `randsplits` simulations
randsplits(model, n) = map(_->randsplits(model), 1:n)

# simple algorithm for α = 0. case
function _randwalk1(splits, clade, ccd)
    isleafclade(clade) && return splits
    csplits = collect(ccd.smap[clade])
    splt = sample(1:length(csplits), Weights(last.(csplits)))
    left = first(csplits[splt])
    rght = clade - left
    push!(splits, (clade, left))
    splits = _randwalk1(splits, left, ccd)
    splits = _randwalk1(splits, rght, ccd)
    return splits
end

# obtain a gene tree from a split set
function treefromsplits(splits::Splits{T}, names) where T
    nodes = Dict{T,DefaultNode{T}}()
    for (γ, δ) in splits
        p, l, r = map(c->_getnode!(nodes, names, c), [γ, δ, γ-δ])
        push!(p, l, r)   
    end
    return getroot(nodes[splits[end][1]])
end

# helper functon for treefromsplits
function _getnode!(nodes, names, n)
    isleafclade(n) && return Node(n, n=names[n])
    haskey(nodes, n) && return nodes[n]
    nodes[n] = Node(n)
    return nodes[n]
end

# draw a random tree from the CCD
function randtree(ccd::CCD)
    splits = randsplits(ccd)
    treefromsplits(splits, ccd.lmap)
end

# generic extension of randtree (also works for MSC)
randtree(model, n) = map(_->randtree(model), 1:n)

# XXX: Ideally, we'd check in some way whether the leaf set of the ccd
# and check if splits actually correspond to splits in the ccd, if not
# return -Inf, but that induces an overhead I guess...
# compute the probability of a set of splits
function logpdf(ccd::CCD, splits::Vector{T}) where T<:Tuple
    ℓ = 0.
    for (γ, δ) in splits
        ℓ += logccp(ccd, γ, δ)
    end
    return isnan(ℓ) ? -Inf : ℓ
end

# compute the probability mass of a single tree under the CCD
function logpdf(ccd::CCD, tree::Node)
    ℓ, _ = _lwalk(tree, ccd, 0.)
    return isnan(ℓ) ? -Inf : ℓ
end

function _lwalk(n::Node, ccd, ℓ)
    isleaf(n) && return ℓ, ccd.lmap[name(n)]
    ℓ, left = _lwalk(n[1], ccd, ℓ) 
    ℓ, rght = _lwalk(n[2], ccd, ℓ)
    δ = left < rght ? left : rght
    γ = left + rght
    ℓ += logccp(ccd, γ, δ) 
    return ℓ, γ
end

# for a vector of trees
function logpdf(ccd::CCD, trees::AbstractVector)
    mapreduce(t->logpdf(ccd, t), +, trees)
end

# for a countmap
function logpdf(ccd::CCD, trees::Dict)
    l = 0.
    for (tree, count) in trees
        l += count*logpdf(ccd, tree)
    end
    return l
end

# compute the marginal likelihood for a bunch of splits on a subset of
# the leaves of a given ccd.
function marginallogpdf(ccd::CCD, splits)
    # XXX assume splits are sorted with rootsplit last! (output of MSC
    # simulation, not MBM simulation!)
    _marginal(ccd, ccd.root, splits, splits[end]...)
end

# γ is the clade in the CCD, c the clade in the data, d the split
# under consideration
function _marginal(ccd, γ, splits, c, d)
    p = 0.
    for δ in keys(ccd.smap[γ])
        q = 0.
        if c ⊂ δ
        elseif c ⊂ (γ - δ)
        end
    end
    return p
end

