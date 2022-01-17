# Arthur Zwaenepoel 2021
# Currently only thought about rooted trees

# Rooted/unrooted? If all input trees are (arbitrarily) rooted at some
# taxon, when α=0, we'll have that all trees from the CCD are rooted
# in the same arbitrary way, and they can be interpreted as draws from
# an unrooted tree distribution. Now when α>0, this is no longer the
# case, and if we don't explicitly distinguish the rooted case from
# the unrooted we'll have different distributions. In the unrooted
# case, represented by an arbitrary rooting, we should have that the
# root split is certain, despite α>1 (this sounds a bit confusing).

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
    CCD

A conditional clade distribution object.
"""
mutable struct CCD{T,V}
    lmap::BiMap{T,String}
    cmap::Dict{T,V}          # clade counts
    smap::Dict{T,Dict{T,V}}  # split counts
    root::T
end

nl(X::CCD) = length(X.lmap)
nc(X::CCD) = length(X.cmap)

function Base.show(io::IO, X::CCD{T}) where T
    write(io, "CCD{$T}(n=$(nl(X)), Γ=$(X.root))")
end

function showclades(X::CCD)
    sort([(k, bitstring(k), v) for (k,v) in X.cmap])
end

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
ccp(ccd::CCD, γ, δ) = ccd.smap[γ][δ] / ccd.cmap[γ]
logccp(ccd, γ, δ) = log(ccd.smap[γ][δ]) - log(ccd.cmap[γ])

# uniform weights for a given vector of elements 
uweights(xs) = fill(1/length(xs), length(xs))

# from a countmap
CCD(trees::AbstractDict; kwargs...) = CCD(collect(trees); kwargs...)

# For a single tree
CCD(tree::Node; kwargs...) = CCD([tree]; kwargs...)

# from a vector of (pairs of) trees
function CCD(trees; lmap=taxonmap(trees[1], UInt16))
    ccd = initccd(lmap)
    for tree in trees
        ccd = addtree!(ccd, tree)
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
    return ccd
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

# check whether a split is present in the (observed part of) a ccd
inccd(ccd, γ, δ) = haskey(ccd.cmap, γ) && haskey(ccd.smap[γ], δ)
inccd(ccd, γ) = haskey(ccd.cmap, γ) && !(ccd.cmap[γ] == 0)

# the number of possible splits of a set of size n
_ns(n) = 2^(n-1) - 1

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

# α > 0. case
# XXX convert to MBM instead!
#function _randwalk2(splits, clade, ccd)
#    isleafclade(clade) && return splits 
#    left = if !inccd(ccd, clade)
#        # uniform random split XXX: this is a bit weird, since
#        # unobserved clades can lead to splits which contain observed
#        # clade, but we don't account for that...
#        splt = randsplit(clade)
#        left = min(splt, clade - splt)
#    else
#        α = getα(ccd, clade)
#        csplits = collect(ccd.smap[clade])
#        observed_splits = first.(csplits)
#        weights = last.(csplits)
#        k = length(csplits)   # number of splits
#        n = cladesize(clade)  
#        N = ccd.cmap[clade]
#        # probability of an observed split
#        denom = log(α * _ns(n) + N)
#        #lpobs = log(k*ccd.α + N) - denom
#        lpobs = log(N) - denom
#        if log(rand()) < lpobs 
#            # observed split
#            splt = sample(1:k, Weights(weights))
#            left = observed_splits[splt]
#        else  
#            # unobserved split
#            left = randsplit(clade)
#            #while left ∈ observed_splits  # XXX rejection sampler
#            #    left = randsplit(clade)
#            #end
#            left
#        end
#    end
#    rght = clade - left
#    push!(splits, (clade, left))
#    splits = _randwalk2(splits, left, ccd)
#    splits = _randwalk2(splits, rght, ccd)
#    return splits
#end

# could be more efficient I guess, also requires the number 2^(n-1),
# which becomes prohibitive for large n
#function randsplit(γ::T) where T
#    g = digits(γ, base=2)
#    n = sum(g)
#    x = rand(1:_ns(n))
#    d = digits(x, base=2)
#    tips = [(i-1) for (i,gi) in enumerate(g) if gi == 1]
#    subclade = 0
#    for i=1:length(d)
#        subclade += d[i] * 2^tips[i]
#    end
#    splt = T(subclade)
#    left = min(splt, γ - splt)
#    return left
#end

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
        #ℓ += _splitp(ccd, γ, δ)
        ℓ += logccp(ccd, γ, δ)
    end
    return ℓ
end

# The contribution of a single split to the tree probability
# currently assumes the uniform prior over splits.
# XXX: this should actually replace `logccp`
# XXX: skip this, instead convert to MBM for the probability under a posterior
#function _splitp(ccd, γ, δ)
#    n = cladesize(γ)
#    α = getα(ccd, γ)
#    Z = α * _ns(n)
#    Z += inccd(ccd, γ) ? ccd.cmap[γ] : 0
#    nδ = inccd(ccd, γ, δ) ? ccd.smap[γ][δ] : 0
#    log(α + nδ) - log(Z)
#end

# compute the probability mass of a single tree under the CCD
function logpdf(ccd::CCD, tree::Node)
    ℓ, _ = _lwalk(tree, ccd, 0.)
    return ℓ
end

function _lwalk(n::Node, ccd, ℓ)
    isleaf(n) && return ℓ, ccd.lmap[name(n)]
    ℓ, left = _lwalk(n[1], ccd, ℓ) 
    ℓ, rght = _lwalk(n[2], ccd, ℓ)
    δ = left < rght ? left : rght
    γ = left + rght
    #ℓ += _splitp(ccd, γ, δ) 
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

