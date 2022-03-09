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
            _update!(cmap, smap, clade, x, w)
            return clade
        end
    end
    walk(tree)
end

# NOTE, we assume the unrooted tree is represented as a rooted one!
addunrooted!(ccd::CCD, lmap, tpair) = addunrooted!(ccd, lmap, tpair[1], tpair[2])  

# naive implementation
#function _addunrooted!(ccd::CCD, lmap, tree::Node, w=1)
#    @assert NewickTree.isbifurcating(tree)
#    o = prewalk(tree)
#    m = length(o) - 2
#    for n in o
#        parent(n) == tree && continue
#        t = set_outgroup(n)
#        addtree!(ccd, lmap, t, w/m)
#    end
#end

# Add an unrooted tree to the CCD. Note that with this, we could in principle
# combine rooted and unrooted trees in the CCD, although in practice one will
# rarely have occasion for that. 
# See the sketch in `docs/img/unrooted-ccd.pdf` for the logic.
function addunrooted!(ccd::CCD, lmap, tree::Node, w=1)
    @assert NewickTree.isbifurcating(tree)
    @unpack cmap, smap = ccd
    o = ccd.root
    m = length(prewalk(tree)) - 2  # possible rootings
    function walk(n)
        if isleaf(n)
            leaf = lmap[name(n)] 
            cmap[leaf] += w
            return leaf, 1
        else
            # recurse left and right down the pseudo-rooted tree
            s1, b1 = walk(n[1])
            s2, b2 = walk(n[2])
            # clade below current node in pseudo-rooted tree
            c1 = s1 + s2
            # s3 is the complement of that clade, i.e. the clade defined by the
            # bipartition in the unrooted tree induced by the branch leading to
            # n in the pseudo-rooted tree
            s3 = o - c1
            # now s1, s2, s3 are the three clades defined by an internal node
            # in the unrooted tree. In the rooted trees compatible with this
            # unrooted tree, we can have (s1,s2|c1), (s1,s3|c2), (s2,s3|c3),
            # (s1,c3|o), (s2,c2|o) and (s3,c1|o), the latter three are root
            # splits
            c2 = s1 + s3
            c3 = s2 + s3
            # we now add the implied root splits. Note we only add those root
            # splits implied by putting the root on either of the two daughter
            # branches of the current node in the pseudo-rooted tree, the
            # rooting implied by putting the root on the parent branch (i.e.
            # (s3,c1|o)) will be dealt with in the recursion
            x2 = min(c2, o-c2)
            x3 = min(c3, o-c3)
            _update!(cmap, smap, o, x2, w/m)
            # if we are dealing with the root node of the pseudo-rooted tree,
            # x2 and x3 are the same, so we return here.
            isroot(n) && return c1, b1 + b2
            _update!(cmap, smap, o, x3, w/m)
            # now for the non-root splits, already mentioned above. The main
            # thing here is to properly compute the weights. For every split,
            # we need to add it to the CCD weighted by the number of unrooted
            # trees that contain that split. This is simply the number of
            # different rootings possible in the 'direction' of the outgroup,
            # and happens to be a function of the size of the outgroup clade.
            # Specifically, if we have the tripartition (s1,s2,s3) and we are
            # adding the (s1,s2|c1) split to the CCD, we need to count the
            # number of rootings containing that split, which is `2|s3|-1`
            b3 = cladesize(s3)
            m1 = 2b1 - 1
            m2 = 2b2 - 1
            m3 = 2b3 - 1
            x1 = min(s1, s2)
            x2 = min(s1, s3)
            x3 = min(s2, s3)
            _update!(cmap, smap, c1, x1, (m3/m)*w)
            _update!(cmap, smap, c2, x2, (m2/m)*w)
            _update!(cmap, smap, c3, x3, (m1/m)*w)
            return c1, b1 + b2
        end
    end
    walk(tree)
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
function _update!(m1::Dict{T,V}, m2, y, x, w=zero(V)) where {T,V}
    if !haskey(m1, y) 
        m1[y] = w
    else
        m1[y] += w
    end
    if !haskey(m2, y) 
        m2[y] = Dict(x=>w)
    elseif !haskey(m2[y], x)
        m2[y][x] = w
    else
        m2[y][x] += w
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
function logpdf(ccd, trees::Dict)
    l = 0.
    for (tree, count) in trees
        l += count*logpdf(ccd, tree)
    end
    return l
end


