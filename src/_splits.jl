# Refactor again: let there be one CCD struct, which subsumes the empirical CCD
# and Bayes CCD... The difference is in the split distributions. But what about
# moment/natural parameter distinctions? The empirical CCD doe snot admit a
# natural parameterization, since we cannot represent θ=0 for the categorical.
# Again, this need not be a problem if we have the moment/natural distinction
# at the splits level.


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


# Split distributions
# ===================
# We have two types of split distributions:
# - empirical split distributions
# - posterior split distributions
# The former are a special case of the latter, but do not admit the natural
# parameterization which we use in e.g. EP.

abstract type AbstractSplits{T,V} end

Base.haskey(m::AbstractSplits, δ) = haskey(m.splits, δ)

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
    Splits{T,V}

Empirical split distribution. This is not much more than a dictionary.
"""
struct Splits{T,V} <: AbstractSplits{T,V}
    parent::T
    splits::Dict{T,V}  # split distribution
end

Base.getindex(x::Splits, δ) = haskey(x, δ) ? x.splits[δ] : 0.

function randsplit(rng::AbstractRNG, x::Splits)
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
abstract type BetaSplits{T,V} <: AbstractSplits{T,V} end

struct NatBetaSplits{T,V} <: BetaSplits{T,V}
    parent::T
    refsplit::T  # reference split for 'wrapping' the categorical
    splits::Dict{T,V}  # represented splits
    n::Vector{Int}  # total number of splits of each size  
    k::Vector{Int}  # number of unrepresented splits of each size
    η0::Vector{V}  # parameter for unrepresented splits
end

struct MomBetaSplits{T,V} <: BetaSplits{T,V}
    parent::T
    refsplit::T  # reference split for 'wrapping' the categorical
    splits::Dict{T,V}  # represented splits
    n::Vector{Int}  # total number of splits of each size  
    k::Vector{Int}  # number of unrepresented splits of each size
    η0::Vector{V}  # parameter for unrepresented splits
end

Base.getindex(x::BetaSplits, δ) = haskey(x, δ) ? x.splits[δ] : m.η0[splitsize(m.γ, δ)]

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
    aρ = as[splitsize(γ, ρ)]
    pρ = haskey(d, ρ) ? log(aρ + d[ρ]) : log(aρ)  # unnormalized pr of split ρ
    η0 = log.(as) .- pρ 
    dd = Dict{T,Float64}()
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
    MomBetaSplits(x.γ, x.refsplit, d, x.n, x.k, η0/Z)
end

function tonatural(x::MomBetaSplits)
    ρ = x.refsplit
    pρ = log(x[ρ])
    η0 = log.(x.η0) .- pρ
    d = Dict(k=>log(p) - pρ for (k,p) in x.splits)
    NatBetaSplits(x.γ, ρ, d, x.n, x.k, η0)
end


# Sampling for the BetaSplits models
# ----------------------------------
# I guess it is good practice to pass the random number generator explicitly,
# although I have never actually used the feature of passing around the RNG in
# simulation code...

function randsplit(rng::AbstractRNG, n::MomBetaSplits)
    length(x.splits) == 0 && return _priorsplit(x)
    splitps = collect(x.splits)
    pr = [w - x.η0[splitsize(x.γ, δ)] for (δ, w) in splitps]
    if rand() < sum(pr)
        i = sample(rng, Weights(pr))
        return first(splitps[i])
    else 
        k = sample(rng, Weights(x.η0 .* x.n))
        return randsplitofsize(rng, x.γ, k)
    end
end

function randsplit(rng::AbstractRNG, x::NatBetaSplits)
    η0 = exp.(x.η0)
    vs = exp.(values(x.splits))
    δs = collect(keys(x.splits))
    pr = x.k .* η0
    Z  = sum(vs) + sum(pr)
    vs ./= Z
    η0 ./= Z
    r = rand(rng) 
    i = 1
    # sort δs, vs by vs?
    while i <= length(δs)
        pδ = vs[i]
        r -= (pδ - η0[splitsize(x.γ, δs[i])])
        r < 0. && return δs[i]
        i += 1
    end
    k = sample(rng, Weights(pr))
    return randsplitofsize(rng, x.γ, k)
end


# Split counts
# ============
# The split counts constitute a sufficient statistic for any CCD. It is
# therefore useful both for preprocessing tree data and for computing
# likelihoods etc. to have a struct which collects this sufficient statistic
# and computes it efficiently for various kinds of input data.
# Note that we call it counts, but in the case the input is weighted in any
# way, these need not be integers.

struct SplitCounts{T,V}
    smap::Dict{T,Dict{T,V}}
    root::T
end

function SplitCounts(trees, m::AbstractDict{T}, weights::AbstractVector{V};
                     rooted=true) where {T,V}
    d = Dict{T,Dict{T,V}}()
    for (tree, w) in zip(trees, weights)
        rooted ? add_splits!(d, m, tree, w) : add_splits_unrooted!(d, m, tree, w)
    end
    return SplitCounts(d, rootclade(m))
end

function SplitCounts(trees, m::AbstractDict{T}; rooted=true) where T
    d = Dict{T,Dict{T,Int64}}()
    for tree in trees
        rooted ? add_splits!(d, m, tree, 1) : add_splits_unrooted!(d, m, tree, 1)
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
    if !haskey(d, γ) 
        d[γ] = Dict(δ=>w)
    elseif !haskey(d[γ], δ)
        d[γ][δ] = w
    else
        d[γ][δ] += w
    end
end


# CCD
# ===
# Using the above implementation for split distributions, the CCD becomes no
# more then collection of these distributions, indexed by the associated clade.

# The SplitCounts are a sufficient statistic for the CCD. We should admit
# computing the likelihood from an individual split set, an individual tree and
# a SplitCounts object.

struct CCD{T,S}
    smap::Dict{T,S}
    root::T
end

Base.haskey(x::CCD, γ) = haskey(x.smap, γ)
Base.haskey(x::CCD, γ, δ) = haskey(x.smap, γ) && haskey(x.smap[γ], δ)
Base.getindex(x::CCD, γ) = x.smap[γ] 
Base.getindex(x::CCD, γ, δ) = haskey(x, γ, δ) ? x.smap[γ][δ] : 0
Base.show(io::IO, x::CCD) = write(io, "CCD(Γ=$(x.root))")

# For the empirical CCD we need this function which 'normalizes' a dictionary
function normalize(x::Dict)
    Z = sum(values(x))
    Dict(k=>v/Z for (k,v) in x)
end

# construct the empirical CCD
function CCD(splits::SplitCounts)
    smap = Dict(γ=>Splits(γ, normalize(x)) for (γ, x) in splits.smap)
    return CCD(smap, splits.root)
end

# Note that it should be possible to preallocate the splits vector, and not
# pushing new splits to it...
function randsplits(rng::AbstractRNG, ccd::CCD{T}) where T
    return _randwalk!(rng, Tuple{T,T}[], ccd.root, ccd)
end

randsplits(ccd::CCD) = randsplits(Random.default_rng(), ccd)
randsplits(ccd::CCD, n::Int) = randsplits(Random.default_rng(), ccd, n)
randsplits(rng, ccd::CCD, n::Int) = map(_->randsplits(rng, ccd), 1:n)

function _randwalk!(rng::AbstractRNG, splits, clade, ccd)
    isleafclade(clade) && return splits
    left = randsplit(rng, ccd[clade])
    rght = clade - left
    push!(splits, (clade, left))
    splits = _randwalk!(rng, splits, left, ccd)
    splits = _randwalk!(rng, splits, rght, ccd)
    return splits
end



