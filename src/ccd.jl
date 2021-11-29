# Arthur Zwaenepoel 2021
# Currently only thought about rooted trees
#
# Depending on our application, different representations are
# desirable. To query probabilities for arbitrary tree topologies,
# we'd need a lot of dictionary-like lookup utilities, while in ALE we
# only need the clades ordered and for each clade easy access to its
# possible splits, so that vector based representations are probably
# optimal.

# TODO: Better use DefaultDict, so that, a.o. logpdf does not break
# when a clade is not present in a tree (but just return -Inf)
# perhaps don't do this after all, but include optional regularization
# (i.e. assume a Dirichlet(α) prior for the categorical tree distribution)
"""
    CCD

A conditional clade distribution object.
"""
mutable struct CCD{T,V}
    leaves::Vector{String}
    lmap  ::Dict{String,T}
    cmap  ::Dict{T,V}          # clade counts
    smap  ::Dict{T,Dict{T,V}}  # split counts
    α     ::Float64
end

nl(X::CCD) = length(X.leaves)
nc(X::CCD) = length(X.cmap)

function Base.show(io::IO, X::CCD{T}) where T
    write(io, "CCD{$T}(n=$(nl(X)), Γ=$(nc(X)))")
end

function showclades(X::CCD)
    sort([(k, bitstring(k), v) for (k,v) in X.cmap])
end

# check if some bitclade represents a leaf
isleafclade(clade) = count_ones(clade) == 1

# get the number of leaves in a clade
cladesize(clade) = count_ones(clade)

# obtain the clade leaf names from its Int representation
getclade(ccd::CCD, clade) = ccd.leaves[Bool.(digits(clade, base=2, pad=nl(ccd)))]

# conditional clade probability
ccp(ccd::CCD, γ, δ) = ccd.smap[γ][δ] / ccd.cmap[γ]
logccp(ccd, γ, δ) = log(ccd.smap[γ][δ]) - log(ccd.cmap[γ])

# uniform weights for a given vector of elements 
uweights(xs) = fill(1/length(xs), length(xs))

# from a vector of trees
function CCD(trees; weights=uweights(trees), α=0., T=UInt16)
    ccd = initccd(trees[1], T, α)
    for (tree, weight) in zip(trees, weights)
        ccd = addtree!(ccd, tree, weight)
    end
    return ccd
end

# from a proportionmap
CCD(trees::Dict; kwargs...) = 
    CCD(collect(keys(trees)), weights=collect(values(trees)); kwargs...)

# For a single tree
CCD(tree::Node; kwargs...) = CCD([tree]; kwargs...)

function initccd(tree::Node, T=UInt16, α=0.)
    lmap = leafclades(tree, T)
    cmap = Dict{T,Float64}(l=>0. for (_,l) in lmap)
    smap = Dict{T,Dict{T,Float64}}()
    kv = collect(lmap)
    leaves = first.(sort(kv, by=last))
    ccd = CCD(leaves, lmap, cmap, smap, α)
end

function leafclades(tree, T=UInt16)
    l = name.(getleaves(tree))
    d = Dict(k=>one(T) << T(i-1) for (i,k) in enumerate(l))
    return d
end

function addtree!(ccd::CCD, tree, w=1.)
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
            x = left < rght ? left : rght
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
inccd(ccd, γ) = haskey(ccd.cmap, γ)

# root all trees identically
function rootall!(trees)
    tree = first(trees)
    leaf = name(first(getleaves(tree)))
    rootall!(trees, leaf)
end

function rootall!(trees, leaf)
    f = x->NewickTree.set_outgroup!(x, leaf)
    map(f, trees)
end

# draw a tree from the ccd, simulates a tree as a set of splits
# XXX This is only correct for α=0.
function randsplits(ccd::CCD{T}) where T
    root = maximum(keys(ccd.cmap))
    return ccd.α == 0. ? 
        _randwalk1(Tuple{T,T}[], root, ccd) : 
        _randwalk2(Tuple{T,T}[], root, ccd)
end

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

function _randwalk2(splits, clade, ccd)
    isleafclade(clade) && return splits
    # XXX here the α mixture approach is quite awkward?
    if rand() < ccd.α || !inccd(ccd, clade)
        splt = randsplit(clade)
        left = min(splt, clade - splt)
        rght = clade - left
    else
        csplits = collect(ccd.smap[clade])
        splt = sample(1:length(csplits), Weights(last.(csplits)))
        left = first(csplits[splt])
        rght = clade - left
    end
    push!(splits, (clade, left))
    splits = _randwalk2(splits, left, ccd)
    splits = _randwalk2(splits, rght, ccd)
    return splits
end

# could be more efficient I guess, also requires the number 2^(n-1),
# which becomes prohibitive for large n
function randsplit(γ::T) where T
    g = digits(γ, base=2)
    n = sum(g)
    x = rand(1:(2^(n-1)-1))
    d = digits(x, base=2)
    tips = [(i-1) for (i,gi) in enumerate(g) if gi == 1]
    subclade = 0
    for i=1:length(d)
        subclade += d[i] * 2^tips[i]
    end
    return T(subclade)
end

# alias
DefaultNode{T} = Node{T,NewickData{Float64,String}}

# obtain a gene tree from a split set
function treefromsplits(splits::Splits{T}, names::Dict) where T
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
    names = Dict(v=>k for (k,v) in ccd.lmap)
    treefromsplits(splits, names)
end

# generic extension of randtree (also works for MSC)
randtree(model, n) = map(_->randtree(model), 1:n)

# XXX: Ideally, we'd check in some way whether the leaf set of the ccd
# and provided splits actually corresponds, if not return -Inf, but
# that induces an overhead I guess...
# compute the probability of a set of splits
function logpdf(ccd::CCD, splits::Vector{T}) where T<:Tuple
    ℓ = 0.
    for (γ, δ) in splits
        # prior contribution
        ℓ += _splitp(ccd, γ, δ)
    end
    return ℓ
end

# prior probability under uniform over splits
_priorp(γ) = 2^(count_ones(γ) - 1) - 1

# The contribution of a single split to the tree probability
# currently assumes the uniform prior over splits.
function _splitp(ccd, γ, δ)
    p = log(ccd.α) - log(_priorp(γ))
    #p = ccd.α/(2.0^(n-1) - 1.0)
    if inccd(ccd, γ, δ)
        p = logaddexp(p, log(1-ccd.α) + logccp(ccd, γ, δ))
        #p += (1.0-ccd.α)*ccp(ccd, γ, δ)
    end
    return p
end

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
    ℓ += _splitp(ccd, γ, δ) 
    return ℓ, γ
end

function logpdf(ccd::CCD, trees::AbstractVector)
    mapreduce(t->logpdf(ccd, t), +, trees)
end

# get clades as bitstrings
function getcladesbits(tree, T=UInt16)
    i = -1  # leaf counter
    clades = T[]
    function walk(n)
        clade = if isleaf(n)
            i += 1
            one(T) << i
        else
            walk(n[1]) + walk(n[2])
        end
        push!(clades, clade)
        return clade
    end
    walk(tree)
    return clades
end

# XXX the above does not work as a hashing function! isomorphic trees
# (same topology, different labels) will have the same clade set!
# This does lead to the following handy function
isisomorphic(t1, t2) = hash(sort(getcladesbits(t1))) == hash(sort(getcladesbits(t2)))

# allows to count topologies using `countmap`
# note though that this is not really worthwhile, reading in the ccd
# tree by tree from the sample appears to be more efficient
Base.hash(tree::Node) = hash(sort(getclades(tree)))
Base.isequal(t1::Node, t2::Node) = hash(t1) == hash(t2)

function getclades(tree)
    i = -1  # leaf counter
    clades = Vector{String}[]
    function walk(n)
        clade = if isleaf(n)
            [name(n)]
        else
            sort(vcat(walk(n[1]), walk(n[2])))
        end
        push!(clades, clade)
        return clade
    end
    walk(tree)
    return clades
end

"""
    kldiv(p::CCD, q::CCD)

Compute the (a) KL divergence `d(p||q) = ∑ₓp(x)log(p(x)/q(x))`.

Not sure how exactly we should do this. For each clade compute the
kldivergence for its split distribution, and weight these
kldivergences by the probability that a tree contains the clade?
"""
function kldiv(p::CCD, q::CCD)  
    D = 0. 
    for (γ, dict) in p.smap
        d = 0.
        p0 = log(_priorp(γ))
        qprior = log(q.α) + p0
        pprior = log(p.α) + p0
        for δ in keys(dict)
            qx = qprior
            if inccd(q, γ, δ) 
                qx = logaddexp(qx, log(1-q.α) + logccp(q, γ, δ))
            end
            px = logaddexp(pprior, log(1-p.α) + logccp(p, γ, δ))
            d += exp(px)*(px - qx)
        end
        D += d*p.cmap[γ]
    end
    return D
end

# symmetrized KL divergence
symmkldiv(p, q) = kldiv(p, q) + kldiv(q, p)
