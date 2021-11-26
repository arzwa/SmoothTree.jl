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
    leafindex::Dict{String,V}
    init     ::Dict{V,Vector{T}}
end

# construct MSC object
function MSC(tree) 
    tree.data.distance = Inf  # always
    leaves = getleaves(tree)
    leafindex = Dict(name(n)=>id(n) for n in leaves)
    # default initialization
    ls = sort(id.(leaves))
    init = Dict(v=>typeof(v)[2^(i-1)] for (i,v) in enumerate(ls))
    MSC(tree, leafindex, init)
end

# adapt an existing MSC model to a new gene family
(model::MSC)(ccd::CCD) = MSC(model.tree, model.leafindex, initmsc(model, ccd))

# initialize the leaf genes for an MSC model for a gene family
function initmsc(model, ccd::CCD{T}) where T
    @unpack leafindex = model
    d = Dict(v=>T[] for (k, v) in model.leafindex)
    for (g, γ) in ccd.lmap
        push!(d[leafindex[_spname(g)]], γ)
    end
    return d
end

# do n `randsplits` simulations
randsplits(model::MSC, n) = map(_->randsplits(model), 1:n)

# generate a tree from the MSC model, in the form of a set of splits.
function randsplits(model::MSC{T}) where T
    _, splits = _coalsplits(model.tree, Tuple{T,T}[], model.init)
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

# define an alias
const Splits{T} = Vector{Tuple{T,T}}

# construct a CCD object from a set of splits
function CCD(model, splits::Vector{Splits{T}}; weights=uweights(splits)) where T
    ccd = initccd(model, splits[1])
    for (s, weight) in zip(splits, weights)
        addsplits!(ccd, s, weight)
    end
    return ccd
end

function initccd(model::MSC, splits::Splits{T}) where T
    leaves = collect(keys(model.leafindex))
    lmap = Dict{String,T}()
    # we make sure it works when a species has multiple leaves
    for (k,v) in model.leafindex
        for (i,g) in enumerate(model.init[v])
            gname = "$(k)_$i"
            lmap[gname] = g
        end
    end
    cmap = Dict(l=>0. for (_,l) in lmap)
    smap = Dict{T,Dict{T,Float64}}()
    CCD(leaves, lmap, cmap, smap, 0)
end

function addsplits!(ccd, s, w=1.)
    for (γ, δ) in s
        _update!(ccd.cmap, ccd.smap, γ, δ)
        ccd.cmap[γ] += w
        ccd.smap[γ][δ] += w
    end
    ccd.total += 1
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

function setdistance!(S, θ::Number)
    for (i,n) in enumerate(postwalk(S))
        isroot(n) && return
        n.data.distance = θ
    end
end
