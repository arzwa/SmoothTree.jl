# we want to simulate lightning fast, and evaluate the probability of
# the simulated tree under the ccd also lightning fast.
# We will be fastest if we simulate a collection of clades and splits
# directly under the MSC. 
# - We initialize at leaves of S with the int representation of the
#   relevant genes
# - We simulate the MSC branch by branch, upon coalescence of c1 and c2
#   we add to the collection of splits (c1+c2, min(c1,c2)), and
#   simulate further with c1+c2
# - For this collection of (γ, δ) pairs we can evaluate the
#   probability under the CCD, without needing recursion
# The crucial thing here is that we need not actually store the tree
# data structure in the way we usually do, and gain efficiency that
# way.

"""
    MSC(species_tree)
 
Multi-species coalescent distribution over gene trees for a given
species tree with branch lengths in coalescent units.
"""
struct MSC{T,V,Ψ}
    tree     ::Ψ
    leafindex::Dict{String,V}     # species name to species tree node
    init     ::Dict{V,Vector{T}}  # species tree node to gene tree leaves
    names    ::Dict{V,String}     # gene tree leaf names
end

# construct MSC object
function MSC(tree) 
    tree.data.distance = Inf  # always
    leaves = getleaves(tree)
    leafindex = Dict(name(n)=>id(n) for n in leaves)
    # default initialization
    leafids = id.(leaves)
    T = eltype(leafids)
    order = sortperm(leafids)
    init  = Dict(v=>T[2^(i-1)] for (i,v) in enumerate(leafids[order]))
    names = Dict(T(2^(i-1))=>name(l) for (i,l) in enumerate(leaves[order]))
    MSC(tree, leafindex, init, names)
end

# adapt an existing MSC model to a new gene family
(model::MSC)(ccd::CCD) = MSC(model.tree, model.leafindex, initmsc(model, ccd)...)

# initialize the leaf genes for an MSC model for a gene family
function initmsc(model, ccd::CCD{T}) where T
    @unpack leafindex = model
    d = Dict(v=>T[] for (k, v) in model.leafindex)
    n = Dict{T,String}()
    for (g, γ) in ccd.lmap
        push!(d[leafindex[_spname(g)]], γ)
        n[γ] = g
    end
    return d, n
end

# utilities for setting species tree branch lengths
n_internal(S) = length(postwalk(S)) - length(getleaves(S)) - 1

function setdistance!(S, θ::Vector)
    for (i,n) in enumerate(postwalk(S))
        isroot(n) && return
        n.data.distance = θ[i]
    end
end

function setdistance_internal!(S, θ::Vector)
    i = 1
    for n in postwalk(S)
        (isroot(n) || isleaf(n)) && continue
        n.data.distance = θ[i]
        i += 1
    end
end

# when dealing with outgroup rooted trees, there is no information
# about branch lengths for the branches coming from the root
function setdistance_internal_rooted!(S, θ::Vector)
    i = 1
    for n in postwalk(S)
        (isroot(n) || isleaf(n)) && continue
        if isroot(parent(n))
            n.data.distance = Inf
        else
            n.data.distance = θ[i]
            i += 1
        end
    end
end

function setdistance!(S, θ::Number)
    for (i,n) in enumerate(postwalk(S))
        isroot(n) && return
        n.data.distance = θ
    end
end

# define an alias
const Splits{T} = Vector{Tuple{T,T}}

# do n `randsplits` simulations
randsplits(model::MSC, n) = map(_->randsplits(model), 1:n)

# generate a tree from the MSC model, in the form of a set of splits.
function randsplits(model::MSC{T}) where T
    _, splits = _coalsplits(model.tree, Splits{T}(), model.init)
    return splits
end

# recursively sample from the MSC, storing the splits
function _coalsplits(n, splits, states)
    isleaf(n) && return _censoredcoalsplits!(splits, distance(n), states[id(n)])
    left, splits = _coalsplits(n[1], splits, states)
    rght, splits = _coalsplits(n[2], splits, states)
    lineages = vcat(left, rght)
    return _censoredcoalsplits!(splits, distance(n), lineages)
end
    
# simulate a censored Kingman coalescent and store splits  
function _censoredcoalsplits!(splits, t, lineages)
    k = length(lineages)
    k <= 1 && return lineages, splits
    t -= randexp() * 2 / (k*(k-1))
    while t > 0. 
        shuffle!(lineages)
        c1 = lineages[end-1]
        c2 = lineages[end]
        γ = c1 + c2
        δ = c1 < c2 ? c1 : c2 
        push!(splits, (γ, δ))
        lineages = [lineages[1:end-2] ; γ]
        k = length(lineages)
        t -= randexp() * 2 / (k*(k-1))
    end
    return lineages, splits
end

# construct a CCD object from a set of splits
function CCD(model, splits::Vector{Splits{T}}; weights=uweights(splits), α=0.) where T
    ccd = initccd(model, splits[1], α)
    for (s, weight) in zip(splits, weights)
        addsplits!(ccd, s, weight)
    end
    return ccd
end

# Initialize a CCD object for an MSC model. This could almost but not
# quite be shared with the usual CCD constructor based on (gene)
# trees, the difference being that here we have to make sure it works
# when a species has multiple leaves.
function initccd(model::MSC, splits::Splits{T}, α=0.) where T
    leaves = collect(keys(model.leafindex))
    lmap = Dict{String,T}()
    for (k,v) in model.leafindex
        for (i,g) in enumerate(model.init[v])
            gname = "$(k)_$i"
            lmap[gname] = g
        end
    end
    cmap = Dict(l=>0. for (_,l) in lmap)
    smap = Dict{T,Dict{T,Float64}}()
    CCD(leaves, lmap, cmap, smap, α)
end

# add a bunch of splits to a CCD
function addsplits!(ccd, s, w=1.)
    for (γ, δ) in s
        _update!(ccd.cmap, ccd.smap, γ, δ)
        ccd.cmap[γ] += w
        ccd.smap[γ][δ] += w
    end
end

# alias
DefaultNode{T} = Node{T,NewickData{Float64,String}}

# obtain a gene tree from a split set
function treefromsplits(model::MSC, splits::Splits{T}) where T
    nodes = Dict{T,DefaultNode{T}}()
    for (γ, δ) in splits
        p, l, r = map(c->_getnode!(nodes, model, c), [γ, δ, γ-δ])
        push!(p, l, r)   
    end
    return nodes[splits[end][1]]
end

# helper functon for treefromsplits
function _getnode!(nodes, model, n)
    isleafclade(n) && return Node(n, n=model.names[n])
    haskey(nodes, n) && return nodes[n]
    nodes[n] = Node(n)
    return nodes[n]
end

# get a random tree from the MSC, *as a tree data structure*
randtree(model::MSC) = treefromsplits(model, randsplits(model))

# The above all simulates *topologies*, but we may also want functions
# to simulate coalescent histories, i.e. with branch lengths...
# However, such branch lengths do not readily translate to gene tree
# branch lengths, sice then we'd need to define the MSC along a time
# tree, with explicit Ne parameters... This is not however of interest
# in the context of this library except for simulation purposes, and
# simulation methods for such things exist elsewhere (e.g. RevBayes).

