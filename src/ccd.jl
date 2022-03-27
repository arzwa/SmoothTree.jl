# March 2022 refactor. I got more clear about the logical structure of the
# subject, with better terminology, so I wanted to reimplement the code base in
# accordance with that structure. The main change is that most of the details
# are now implemented at the level of the split distributions, since the
# overall structure of a CCD is the same everywhere.

# Clades and splits
# =================
# Here we define some functions for dealing with clades and splits.

# Note that clades are simply represented by integers, whose binary expansion
# is a bitstring recording presence/absence of leaf clades.  In other words the
# clade with id 13 is clade 1101, representing the clade containing leaf 4,
# leaf 3 and leaf 1. 
# This restricts us here (currently) to trees with at most 62 leaves, although
# it would be possible to use larger bits types, or some kind of bitstring
# type.

# check if some bitclade represents a leaf
isleafclade(clade) = count_ones(clade) == 1

# check if cherry clade
ischerry(clade) = count_ones(clade) == 2

# get the number of leaves in a clade
cladesize(clade) = count_ones(clade)

# get the root clade for n leaves
rootclade(n, ::Type{T}=UInt64) where T = T(2^n - 1) 
rootclade(m::BiMap{T}) where T = rootclade(length(m), T)

# the number of possible splits of a set of size n
nsplits(n) = 2^(n-1) - 1

"""
    maxsplit(γ)

The supremum in the order of splits of `γ`, defined to be the clade `(γ - c)`
where `c` is the leaf in `γ` with largest index.
"""
maxsplit(γ::T) where T = T(γ - 2^(ndigits(γ, base=2) - 1))

function splitcherry(γ)
    @assert ischerry(γ)
    c1 = maxsplit(γ)
    return c1, γ-c1
end


# Splits
# ======
# Splits is just a type alias for a vector of tuples.
const DefaultNode{T} = Node{T,NewickData{Float64,String}}

"""
    Splits{T}

A collection of splits, i.e. tuples `(γ, δ)`, so that `δ ⊂ γ` and `δ < γ - δ`. 
"""
const Splits{T} = Vector{Tuple{T,T}}

function getsplits(n::DefaultNode, m::AbstractDict{T,V}) where {T,V}
    splits = Splits{T}()
    _getsplits(splits, n, m)
    return splits
end

function _getsplits(splits, n, m)
    isleaf(n) && return m[name(n)]
    a = _getsplits(splits, n[1], m)
    b = _getsplits(splits, n[2], m)
    push!(splits, (a + b, min(a,b)))
    return a + b
end

# Branches
# ========
# This is a useful alternative representation of a tree object, although I am
# not entirely sure anymore whether it has benefits over a usual recursive tree
# object defined in terms of `NewickTree.Node`.

"""
    Branches{T}

A vector representation of a phylogenetic tree (with `Float64` branch lengths).
"""
struct Branches{T,V}
    splits::Splits{T}
    xs::Vector{V}
end

# undef initializer
function Branches(_::UndefInitializer, T, n) 
    Branches(Splits{T}(undef, n), Vector{Float64}(undef, n))
end

Base.length(b::Branches) = length(b.xs)
Base.iterate(b::Branches) = ((b.splits[1]..., b.xs[1]), 1)
Base.iterate(b::Branches, i) = i >= length(b) ? nothing : ((b.splits[i+1]..., b.xs[i+1]), i+1)
Base.getindex(b::Branches, i) = (b.splits[i]..., b.xs[i])
Base.copy(b::Branches) = Branches(copy(b.splits), copy(b.xs))

function Base.setindex!(b::Branches, v, i)
    b.splits[i] = (v[1], v[2])
    b.xs[i] = v[3]
end

"""
    getbranches(n::Node, m::AbstractDict)
    
Get `Branches` struct from a `NewickTree` tree (`Node`). Note that we assume
the branches are on a rate (ℝ⁺) scale (we log-transform them). The branches
will be pre-ordered.
"""
function getbranches(n::DefaultNode, m::AbstractDict{T}) where {T}
    nn = length(postwalk(n))-1
    branches = Branches(undef, T, nn)
    _getbranches(branches, n, m, 1)
    return branches 
end

function _getbranches(branches, n, m, i)
    isleaf(n) && return m[name(n)], distance(n), i
    a, da, j = _getbranches(branches, n[1], m, i+2)
    b, db, j = _getbranches(branches, n[2], m, j)
    branches[i] = (a + b, a, log(da))
    branches[i+1] = (a + b, b, log(db))
    return a + b, distance(n), j
end

branchdict(b::Branches) = Dict((x1,x2)=>x3 for (x1,x2,x3) in b)


# Split distributions
# ===================
# The split distributions are the main level of importance for the CCD, i.e.
# they define the probability distribution over trees, whereas the CCD-level
# just accounts for the Markovian splitting process that generates actual
# trees.

# We have two main kinds of split distributions:
# - empirical split distributions
# - posterior split distributions
# The former are a special case of the latter, but do not admit the natural
# parameterization which we use in e.g. EP. We will make all split
# distributions subtypes of an abstract type.

abstract type AbstractSplits{T,V} end

Base.haskey(m::AbstractSplits, δ) = haskey(m.splits, δ)
Base.iterate(m::AbstractSplits) = iterate(m.splits)
Base.iterate(m::AbstractSplits, i) = iterate(m.splits, i)
Base.length(m::AbstractSplits) = length(m.splits)
Base.setindex!(m::AbstractSplits, v, δ) = m.splits[δ] = v
Base.keys(m::AbstractSplits) = keys(m.splits)

"""
    randsplit(x<:AbstractSplits)

Simulate a random split according to split distribution `x`.
"""
randsplit(x::AbstractSplits) = randsplit(Random.default_rng(), x)


# Empirical split distribution
# ----------------------------
# Note that in the case of an empirical CCD, we would like to store the clade
# and split counts. Or don't we? Alternatively we may deal with that only in
# the constructor? Or we could only store the clade counts, not split counts?

"""
    EmpiricalSplits{T,V}

Empirical split distribution. This is not much more than a dictionary.
"""
struct EmpiricalSplits{T,V} <: AbstractSplits{T,V}
    parent::T
    splits::Dict{T,V}  # split distribution
end

Base.getindex(x::EmpiricalSplits, δ) = haskey(x, δ) ? x.splits[δ] : 0.

logpdf(x::EmpiricalSplits, δ) = log(x[δ])

function randsplit(rng::AbstractRNG, x::EmpiricalSplits)
    δs = collect(x.splits)
    i = sample(rng, 1:length(δs), Weights(last.(δs)))
    return first(δs[i])
end


# Beta-splitting posterior split distribution
# -------------------------------------------
# For the Beta-splitting posterior CCD, should we have two different types, or
# dispatch on some type parameter?

"""
    MBMSplits

MBMSplits types implement posterior split distributions assuming a Markov
branching model prior for the split distribution (i.e. a prior on subclade
sizes).
"""
abstract type MBMSplits{T,V} <: AbstractSplits{T,V} end

struct NatBetaSplits{T,V} <: MBMSplits{T,V}
    parent::T
    refsplit::T  # reference split for 'wrapping' the categorical
    splits::Dict{T,V}  # represented splits
    n::Vector{Int}  # total number of splits of each size  
    k::Vector{Int}  # number of unrepresented splits of each size
    η0::Vector{V}  # parameter for unrepresented splits
end

struct MomBetaSplits{T,V} <: MBMSplits{T,V}
    parent::T
    refsplit::T  # reference split for 'wrapping' the categorical
    splits::Dict{T,V}  # represented splits
    n::Vector{Int}  # total number of splits of each size  
    k::Vector{Int}  # number of unrepresented splits of each size
    η0::Vector{V}  # parameter for unrepresented splits
end

Base.getindex(x::MBMSplits, δ) = haskey(x, δ) ? 
    x.splits[δ] : x.η0[splitsize(x.parent, δ)]

"""
    prune!(x::MBMSplits, atol)

Prune a split distribution with MBM prior. If the probability of a represented
split is within `atol` of an unrepresented one of the same size, remove it from 
the explicitly represented splits.
"""
function prune!(x::MBMSplits, atol)
    for (δ, v) in x
        s = splitsize(x.parent, δ)
        if isapprox(exp(v), exp(x.η0[s]), atol=atol)
            delete!(x.splits, δ)
            x.k[s] += 1
        end
    end 
    return x
end


"""
    NatBetaSplits(clade, counts, bsd, α)

Get a natural parameter BetaSplits object from split counts `d`, assuming the
Beta-splitting Dirichlet prior `bsd` with pseudo-count α and shape parameter
`β`. 
"""
function NatBetaSplits(γ::T, d, bsd::BetaSplitTree{V}, α) where {T,V}
    ρ  = maxsplit(γ)
    s  = cladesize(γ) 
    ns = nsplits.(s, 1:s÷2) 
    as = α .* bsd.q[s-2]
    aρ = as[splitsize(γ, ρ)]  # pseudocount for the reference/max split
    pρ = haskey(d, ρ) ? log(aρ + d[ρ]) : log(aρ)  # unnormalized pr of split ρ
    η0 = log.(as) .- pρ 
    dd = Dict{T,V}()
    kc = zeros(Int, s÷2)  # counter
    for (δ, count) in d
        k = splitsize(γ, δ)
        dd[δ] = log(as[k] + count) - pρ
        kc[k] += 1
    end
    k = ns .- kc
    NatBetaSplits(γ, ρ, dd, ns, k, η0)
end

function tomoment(x::NatBetaSplits)
    η0 = exp.(x.η0)
    d = Dict(k=>exp(v) for (k,v) in x.splits)
    S = sum(values(d))
    Z = S + sum(x.k .* η0)
    for (k, v) in d
        d[k] /= Z
    end
    MomBetaSplits(x.parent, x.refsplit, d, x.n, x.k, η0/Z)
end

function tonatural(x::MomBetaSplits)
    ρ = x.refsplit
    pρ = log(x[ρ])
    η0 = log.(x.η0) .- pρ
    d = Dict(k=>log(p) - pρ for (k,p) in x.splits)
    NatBetaSplits(x.parent, ρ, d, x.n, x.k, η0)
end


# Probabilities for the BetaSplits models
# ---------------------------------------
logpdf(x::MomBetaSplits, δ) = log(x[δ])

function logpdf(x::NatBetaSplits, δ)
    y = exp.(collect(values(x.splits)))
    Z = sum(y) + sum(x.k .* exp.(x.η0))
    return x[δ] - log(Z)
end


# Sampling for the BetaSplits models
# ----------------------------------
# I guess it is good practice to pass the random number generator explicitly,
# although I have never actually used the feature of passing around the RNG in
# simulation code...

function randsplit(rng::AbstractRNG, x::MomBetaSplits)
    vs = collect(values(x.splits))
    δs = collect(keys(x.splits))
    _randsplit(rng, x.parent, δs, vs, x.η0 .* x.n, x.η0) 
end

function randsplit(rng::AbstractRNG, x::NatBetaSplits)
    η0 = exp.(x.η0)
    pr = x.k .* η0
    vs = exp.(values(x.splits))
    δs = collect(keys(x.splits))
    Z  = sum(vs) + sum(pr)
    vs ./= Z
    η0 ./= Z
    _randsplit(rng, x.parent, δs, vs, pr, η0)
end

function _randsplit(rng::AbstractRNG, γ, δs, ps, pr, η0)
    i = 1
    r = rand(rng) 
    # sort δs, ps by ps?
    while i <= length(δs)
        pδ = ps[i]
        r -= (pδ - η0[splitsize(γ, δs[i])])
        r < 0. && return δs[i]
        i += 1
    end
    k = sample(rng, Weights(pr))
    return randsplitofsize(rng, γ, k)
end


# Split counts
# ============
# The split counts constitute a sufficient statistic for any CCD. It is
# therefore useful both for preprocessing tree data and for computing
# likelihoods etc. to have a struct which collects this sufficient statistic
# and computes it efficiently for various kinds of input data.
# Note that we call it counts, but in the case the input is weighted in any
# way, these need not be integers.
const SplitDict{T,V} = Dict{T,Dict{T,V}}

"""
    SplitCounts(trees, clademap, [weights; rooted=true])

(Weighted) split counts for a collection of trees. This is a sufficient
statistic for CCD distributions over cladograms.
"""
struct SplitCounts{T,V}
    smap::SplitDict{T,V}
    root::T
end

Base.iterate(x::SplitCounts) = iterate(x.smap)
Base.iterate(x::SplitCounts, i) = iterate(x.smap, i)
Base.length(x::SplitCounts) = length(x.smap)

SplitCounts(root::T) where {T<:Integer} = SplitCounts(SplitDict{T,Int64}(), root)
SplitCounts(n::Node, m::AbstractDict) = SplitCounts([n], m)

# some inelegant code repetition here...
SplitCounts(d::AbstractDict, m::AbstractDict) = 
    SplitCounts(collect(keys(d)), m, collect(values(d)))

SplitCountsUnrooted(d::AbstractDict, m::AbstractDict) = 
    SplitCountsUnrooted(collect(keys(d)), m, collect(values(d)))

function SplitCounts(ts, m::AbstractDict{T}, ws::AbstractVector{V}) where {T,V}
    d = SplitDict{T,V}()
    for (tree, w) in zip(ts, ws)
        add_splits!(d, m, tree, w)
    end
    return SplitCounts(d, rootclade(m))
end

function SplitCounts(trees, m::AbstractDict{T}) where T
    d = SplitDict{T,Int64}()
    for tree in trees
        add_splits!(d, m, tree, 1)
    end
    return SplitCounts(d, rootclade(m))
end

function SplitCountsUnrooted(ts, m::AbstractDict{T}, ws::AbstractVector{V}) where {T,V}
    d = SplitDict{T,Float64}()
    for (tree, w) in zip(ts, ws)
        add_splits_unrooted!(d, m, tree, w)
    end
    return SplitCounts(d, rootclade(m))
end

function SplitCountsUnrooted(ts, m::AbstractDict{T}) where T
    d = SplitDict{T,Float64}()
    for tree in ts
        add_splits_unrooted!(d, m, tree, 1.)
    end
    return SplitCounts(d, rootclade(m))
end

function add_splits!(d, m, tree::Node, w=1)
    @assert NewickTree.isbifurcating(tree)
    function walk(n)
        if isleaf(n)
            leaf = m[name(n)] 
            return leaf
        else
            left = walk(n[1])
            rght = walk(n[2])
            clade = left + rght
            x = min(left, rght)
            update!(d, clade, x, w)
            return clade
        end
    end
    walk(tree)
end

function add_splits_unrooted!(d, lmap, tree::Node, w=1)
    @assert NewickTree.isbifurcating(tree)
    o = rootclade(lmap)
    m = length(prewalk(tree)) - 2  # possible rootings
    function walk(n)
        if isleaf(n)
            leaf = lmap[name(n)] 
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
            update!(d, o, x2, w/m)
            # if we are dealing with the root node of the pseudo-rooted tree,
            # x2 and x3 are the same, so we return here.
            isroot(n) && return c1, b1 + b2
            update!(d, o, x3, w/m)
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
            update!(d, c1, x1, (m3/m)*w)
            update!(d, c2, x2, (m2/m)*w)
            update!(d, c3, x3, (m1/m)*w)
            return c1, b1 + b2
        end
    end
    walk(tree)
end

function update!(d, γ, δ::T, w::V) where {T,V}
    # we don't store cherry clades
    ischerry(γ) && return
    if !haskey(d, γ) 
        d[γ] = Dict(δ=>w)
    elseif !haskey(d[γ], δ)
        d[γ][δ] = w
    else
        d[γ][δ] += w
    end
end

function SplitCounts(xs::Vector{Branches{T,V}}, ws::Vector{V}) where {T,V}
    d = SplitDict{T,V}()
    for (x, w) in zip(xs, ws)
        add_splits!(d, x, w)
    end
    return SplitCounts(d, maximum(keys(d)))
end

function add_splits!(d, x::Branches, w)
    for (γ, δ, _) in x
        δ = min(δ, γ-δ)
        update!(d, γ, δ, w)
    end
end

# CCD
# ===
# Using the above implementation for split distributions, the CCD becomes no
# more then collection of these distributions, indexed by the associated clade.

# The SplitCounts are a sufficient statistic for the CCD. We should admit
# computing the likelihood from an individual split set, an individual tree and
# a SplitCounts object.

"""
    CCD(splitcounts[, model])

Conditional clade distribution
"""
struct CCD{T,S,M<:SplittingModel}
    smap ::Dict{T,S}
    prior::M
    root ::T
end

Base.haskey(x::CCD, γ) = haskey(x.smap, γ)
Base.getindex(x::CCD, γ) = x.smap[γ] 
Base.setindex!(x::CCD, v, γ) = x.smap[γ] = v
Base.show(io::IO, x::CCD) = write(io, "CCD(Γ=$(x.root))")
Base.iterate(x::CCD) = iterate(x.smap)
Base.iterate(x::CCD, i) = iterate(x.smap, i)
Base.length(x::CCD) = length(x.smap)

# For the empirical CCD we need this function which 'normalizes' a dictionary
function normalize(x::Dict)
    Z = sum(values(x))
    Dict(k=>v/Z for (k,v) in x)
end

# Construct the empirical CCD
function CCD(splits::SplitCounts, args...)
    smap = Dict(γ=>EmpiricalSplits(γ, normalize(x)) for (γ, x) in splits.smap)
    return CCD(smap, NoModel(), splits.root)
end

# Construct a beta-splitting CCD
function CCD(splits::SplitCounts, bsd::BetaSplitTree, α=1.)
    smap = Dict(γ=>NatBetaSplits(γ, x, bsd, α) 
                for (γ, x) in splits.smap if !ischerry(γ))
    return CCD(smap, bsd, splits.root)
end

tomoment(x::CCD) = CCD(Dict(γ=>tomoment(v) for (γ,v) in x), x.prior, x.root)

function prune!(x::CCD{T,V}, atol) where {T,V}
    clades = Set(x.root)
    empty = Set{T}()
    for (γ, v) in x
        prune!(v, atol)
        # all splits with non-negligible probabilities are to be kept
        # note that we also need to keep the complements, which are
        # not in the split distribution of γ but may have their own
        # split distribution in the smap!
        union!(clades, keys(v.splits))
        union!(clades, γ .- keys(v.splits))  
        length(v) == 0 && union!(empty, γ)
    end
    # those clades nowhere seen will be deleted
    toprune = setdiff(keys(x.smap), clades)  
    for γ in union(toprune, empty)
        delete!(x.smap, γ)
    end
end


# Sampling random cladograms from a CCD
# -------------------------------------
# We do not explicitly store the splits of cherries, since they are invariable.

function randsplits(rng::AbstractRNG, ccd::CCD{T}) where T
    return _randwalk!(rng, Tuple{T,T}[], ccd.root, ccd)
end

randsplits(ccd::CCD) = randsplits(Random.default_rng(), ccd)
randsplits(ccd::CCD, n::Int) = randsplits(Random.default_rng(), ccd, n)
randsplits(rng, ccd::CCD, n::Int) = map(_->randsplits(rng, ccd), 1:n)

randsplit(ccd, γ) = randsplit(Random.default_rng(), ccd, γ)
randsplit(rng::AbstractRNG, ccd, γ) = haskey(ccd, γ) ? 
    randsplit(rng, ccd[γ]) : randsplit(rng, ccd.prior, γ)

function _randwalk!(rng::AbstractRNG, splits, clade, ccd)
    (ischerry(clade) || isleafclade(clade)) && return splits
    left = randsplit(rng, ccd, clade)
    rght = clade - left
    push!(splits, (clade, left))
    splits = _randwalk!(rng, splits, left, ccd)
    splits = _randwalk!(rng, splits, rght, ccd)
    return splits
end

randtree(ccd::CCD, lmap) = randtree(Random.default_rng(), ccd, lmap)
randtree(ccd::CCD, lmap, n) = randtree(Random.default_rng(), ccd, lmap, n)
randtree(rng::AbstractRNG, ccd::CCD, lmap, n) = map(_->randtree(rng, ccd, lmap), 1:n)

function randtree(rng::AbstractRNG, ccd::CCD, lmap)
    splits = randsplits(rng, ccd)
    gettree_addcherry(splits, lmap)
end

# Note that for the CCD, we don't simulate the cherry splits (we could, but it
# is not necessary, and spares some unncessary computation). This leads however
# to a slight modification to get a tree from a set of splits.
function gettree_addcherry(splits::Splits{T}, names) where T
    nodes = Dict{T,DefaultNode{T}}()
    for (γ, δ) in splits
        p, l, r = map(c->_getnode_addcherry!(nodes, names, c), [γ, δ, γ-δ])
        push!(p, l, r)   
    end
    return getroot(nodes[splits[end][1]])
end

function _getnode_addcherry!(nodes, names, n)
    isleafclade(n) && return Node(n, n=names[n])
    haskey(nodes, n) && return nodes[n]
    nodes[n] = Node(n)
    if ischerry(n) 
        c1, c2 = splitcherry(n)
        l = _getnode!(nodes, names, c1)
        r = _getnode!(nodes, names, c2)
        push!(nodes[n], l, r)
    end
    return nodes[n]
end


# Computing probabilities for the CCD
# -----------------------------------
# We should implement `logpdf` function for multiple data structures. (1) A
# single collection of splits, (2) a `SplitCounts` object, (3) a single tree.
# Not sure about (3), perhaps we should enforce going through either (1) or
# (2).

# Get the conditional clade probability
logpdf(ccd::CCD, γ, δ) = haskey(ccd, γ) ? logpdf(ccd[γ], δ) : logpdf(ccd.prior, γ, δ)

function logpdf(ccd::CCD, γ, d::AbstractDict)
    ℓ = 0.
    if haskey(ccd, γ)
        x = ccd[γ]
        for (δ, k) in d
            ℓ += k*logpdf(x, δ)
        end
    else
        for (δ, k) in d
            ℓ += k*logpdf(ccd.prior, γ, δ)
        end
    end
    return ℓ
end

"""
    logpdf(ccd::CCD, splits::Splits)
    logpdf(ccd::CCD, splits::SplitCounts)

Log probability for a single collection of splits.
"""
function logpdf(ccd::CCD, splits)
    ℓ = 0.
    for (γ, δ) in splits
        ℓ += logpdf(ccd, γ, δ)
    end
    return isnan(ℓ) ? -Inf : ℓ
end

"""
    logpartition(x::CCD)

Compute the log-partition function for `x`.

The log-partition function of a categorical distribution on k categories with
moment parameter `θ = (θ1, θ2, …, θ{k-1})` is `-log(1-∑i θi) = -log θk`. The
CCD defines a categorical distribution on tree topologies. We have defined an
order on trees (in particular we have a well-defined last tree, see `reftree`).
So it appears we can easily compute -log θk. 
"""
logpartition(x::CCD) = -logpdf(x, reftree(x.root)) 

# get the last tree in the induced tree order
function reftree(x::T) where T
    m = cladesize(x)
    splits = Splits{T}(undef, m-1)
    for k=m-1:-1:1
        y = rootclade(k, T)
        splits[m-k] = (x, y)
        x = y
    end
    return splits
end

# this one uses `maxsplit` recursively, but for our current definition of
# maxsplit, the above is more elegant and efficient
#function reftree2(x::T) where T
#    splits = Tuple{T,T}[]
#    function walk(x)
#        isleafclade(x) && return 
#        a = maxsplit(x)
#        b = x - a
#        push!(splits, (x, min(a,b)))
#        walk(a)
#        walk(b)
#    end
#    walk(x)
#    return splits
#end

# should have ccps as support values?
function maptree(x::CCD, spmap)
    n = Node(x.root)
    _maptree(x, x.root, n, spmap)
end

function _maptree(x::CCD, γ, n, spmap)
    if isleafclade(γ)
        n.data.name = spmap[γ]
        return n
    elseif ischerry(γ)
        c1, c2 = splitcherry(γ)
        push!(n, Node(c1, n=spmap[c1]))
        push!(n, Node(c2, n=spmap[c2]))
        return n
    else
        δ = first(argmax(last, x[γ].splits))
        s = exp(logpdf(x, γ, δ))
        n.data.name = string(round(Int, s*100))
        n.data.support = s
        c1 = Node(δ)
        c2 = Node(γ-δ)
        push!(n, c1, c2)
        _maptree(x, δ, c1, spmap)
        _maptree(x, γ-δ, c2, spmap)
        return n
    end 
end

mapsplits(x::CCD{T}) where T = _mapsplits!(Splits{T}(), x, x.root)

function _mapsplits!(splits, x, γ)
    (isleafclade(γ) || ischerry(γ)) && return
    δ = first(argmax(last, x[γ].splits))
    push!(splits, (γ, δ))
    _mapsplits!(splits, x, δ)
    _mapsplits!(splits, x, γ-δ)
    return splits
end




