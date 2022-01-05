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

# XXX Minimalist implementation
"""
    MSC(species_tree)
 
Multi-species coalescent distribution over gene trees for a given
species tree with branch lengths in coalescent units.
"""
struct MSC{T,Ψ}
    tree::Ψ
    init::Dict{T,Vector{T}}  # species tree node to gene tree leaves
    function MSC(tree::Ψ, init::Dict{T,Vector{T}}) where {Ψ,T}
        tree.data.distance = Inf  # always
        new{T,Ψ}(tree, init)
    end
end

# default initialization
default_init(S, tmap::BiMap) = Dict(id(n)=>[tmap[name(n)]] for n in getleaves(S))

MSC(S, tmap::BiMap) = MSC(S, default_init(S, tmap))

# generate a tree from the MSC model, in the form of a set of splits.
function randsplits(model::MSC{T}) where T
    _, splits = _coalsplits(model.tree, Splits{T}(), model.init)
    return splits
end

# TODO, write non-recursive simulator
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
        c1 = pop!(lineages)
        c2 = pop!(lineages)
        γ = c1 + c2
        δ = c1 < c2 ? c1 : c2 
        push!(splits, (γ, δ))
        push!(lineages, γ)
        k = length(lineages)
        t -= randexp() * 2 / (k*(k-1))
    end
    return lineages, splits
end

# get a random tree from the MSC, *as a tree data structure*
randtree(model::MSC, lmap::AbstractDict) = treefromsplits(randsplits(model), lmap)
randtree(model::MSC, lmap::AbstractDict, n) = map(_->randtree(model, lmap), 1:n)

