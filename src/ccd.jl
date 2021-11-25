# Arthur Zwaenepoel 2021
# Currently only thought about rooted trees
#
# Depending on our application, different representations are
# desirable. To query probabilities for arbitrary tree topologies,
# we'd need a lot of dictionary-like lookup utilities, while in ALE we
# only need the clades ordered and for each clade easy access to its
# possible splits, so that vector based representations are probably
# optimal.

# Better use DefaultDict, so that, a.o. logpdf does not break when a
# clade is not present in a tree (but just return -Inf)
mutable struct CCD{T,V}
    leaves::Vector{String}
    lmap  ::Dict{String,T}
    cmap  ::Dict{T,V}          # clade counts
    smap  ::Dict{T,Dict{T,V}}  # split counts
    total ::Int64
end

nl(X::CCD) = length(X.leaves)
nc(X::CCD) = length(X.cmap)
nt(X::CCD) = X.total

function Base.show(io::IO, X::CCD{T}) where T
    write(io, "CCD{$T}(n=$(nl(X)), Γ=$(nc(X)), N=$(nt(X)))")
end

function showclades(X::CCD)
    sort([(k, bitstring(k), v) for (k,v) in X.cmap])
end

# check if some bitclade represents a leaf
isleafclade(clade) = count_ones(clade) == 1

# obtain the clade leaf names from its Int representation
getclade(ccd::CCD, clade) = ccd.leaves[Bool.(digits(clade, base=2, pad=nl(ccd)))]

# conditional clade probability
ccp(ccd::CCD, γ, δ) = ccd.smap[γ][δ] / ccd.cmap[γ]

# uniform weights for a given vector of elements 
uweights(xs) = fill(1/length(xs), length(xs))

# from a vector of trees
function CCD(trees; weights=uweights(trees), T=UInt16)
    ccd = initccd(trees[1], T)
    for (tree, weight) in zip(trees, weights)
        ccd = addtree!(ccd, tree, weight)
    end
    return ccd
end

# from a proportionmap
CCD(trees::Dict; T=UInt16) = 
    CCD(collect(keys(trees)), weights=collect(values(trees)), T=T)

# For a single tree
CCD(tree::Node; T=UInt16) = CCD([tree], T=T)

function initccd(tree, T=UInt16)
    lmap = leafclades(tree, T)
    cmap = Dict{T,Float64}(l=>0. for (_,l) in lmap)
    smap = Dict{T,Dict{T,Float64}}()
    kv = collect(lmap)
    leaves = first.(sort(kv, by=last))
    ccd = CCD(leaves, lmap, cmap, smap, 0)
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
    ccd.total += 1
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

# draw a tree from the ccd
function randtree(ccd::CCD{T}) where T
    root = maximum(keys(ccd.cmap))
    return _randwalk(T[], root, ccd)
end

function _randwalk(clades, clade, ccd)
    push!(clades, clade)
    isleafclade(clade) && return clades
    splits = collect(ccd.smap[clade])
    splt = sample(1:length(splits), Weights(last.(splits)))
    left = first(splits[splt])
    rght = clade - left
    clades = _randwalk(clades, left, ccd)
    clades = _randwalk(clades, rght, ccd)
    return clades
end

# compute the probability of a set of splits
function logpdf(ccd::CCD, splits::Vector{T}) where T<:Tuple
    ℓ = 0.
    for (γ, δ) in splits
        (!haskey(ccd.cmap, γ) || !haskey(ccd.smap[γ], δ)) && return -Inf
        ℓ += log(ccd.smap[γ][δ]) - log(ccd.cmap[γ])
    end
    return ℓ
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
    (!haskey(ccd.cmap, γ) || !haskey(ccd.smap[γ], δ)) && return -Inf, γ
    ℓ += log(ccd.smap[γ][δ]) - log(ccd.cmap[γ])
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

