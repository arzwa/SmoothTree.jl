# we want to simulate lightning fast, and evaluate the probability of the
# simulated tree under the ccd also lightning fast.  We will be fastest if we
# simulate a collection of clades and splits directly under the MSC. 
# - We initialize at leaves of S with the int representation of the relevant
#   genes
# - We simulate the MSC branch by branch, upon coalescence of c1 and c2 we add
#   to the collection of splits (c1+c2, min(c1,c2)), and simulate further with
#   c1+c2
# - For this collection of (γ, δ) pairs we can evaluate the probability under
#   the CCD, without needing recursion
# The crucial thing here is that we need not actually store the tree data
# structure in the way we usually do, and gain efficiency that way.

# Minimalist implementation
"""
    MSC(species_tree, initialization::Dict)
 
Multi-species coalescent distribution over gene trees for a given
species tree with branch lengths in coalescent units. 

The `initialization` is a dictionary recording for each leaf node of
the species tree a vector of genes, which should be clade labels (i.e.
powers of 2) (see `taxonmap` and the example below).

# Example
```julia-repl
julia> tree = nw"((F,(G,H):1):1,((B,((A,C):1,(D,E):1):1):1,O):1);";
       m = taxonmap(tree);
       init = Dict(id(n)=>[m[name(n)]] for n in getleaves(tree))
Dict{UInt16, Vector{UInt16}} with 9 entries:
  0x0005 => [0x0002]
  0x000d => [0x0020]
  0x0006 => [0x0004]
  0x000f => [0x0040]
  0x0010 => [0x0080]
  0x0009 => [0x0008]
  0x000c => [0x0010]
  0x0011 => [0x0100]
  0x0003 => [0x0001]

julia> M = SmoothTree.MSC(tree, init);
       randtree(M, m)
((F,(D,E)),((G,H),((B,(A,C)),O)));
```
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

# recursively sample from the MSC, storing the splits
function _coalsplits(n, splits, states)
    isleaf(n) && return _censoredcoalsplits!(splits, distance(n), states[id(n)])
    left, splits = _coalsplits(n[1], splits, states)
    rght, splits = _coalsplits(n[2], splits, states)
    lineages = vcat(left, rght)
    return _censoredcoalsplits!(splits, distance(n), lineages)
end
    
# simulate a censored Kingman coalescent and store splits  
# XXX this modifies `lineages`! This could mess with our initialization
# dictionaries when dealing with multiple lineages in a leaf branch if we're
# not careful
function _censoredcoalsplits!(splits, t, lineages)
    k = length(lineages)
    k <= 1 && return lineages, splits
    t -= randexp() * 2 / (k*(k-1))
    while t > 0. 
        shuffle!(lineages)
        c1 = pop!(lineages)
        c2 = pop!(lineages)
        γ = c1 + c2
        δ = min(c1, c2) 
        push!(splits, (γ, δ))
        push!(lineages, γ)
        k = length(lineages)
        t -= randexp() * 2 / (k*(k-1))
    end
    return lineages, splits
end

# get a random tree from the MSC, *as a tree data structure*
randtree(model::MSC, lmap::AbstractDict) = gettree(randsplits(model), lmap)
randtree(model::MSC, lmap::AbstractDict, n) = map(_->randtree(model, lmap), 1:n)

# non-recursive version -- takes the same amount of time and has exactly the
# same amount of allocations... (recursive version seems even to have a slight
# edge, because it doesn't have to store the states at internal nodes I guess)
#function randsplits2(model::MSC{T}, order) where T
#    @unpack tree, init = model
#    splits = Splits{T}()
#    for n in order
#        isleaf(n) && continue
#        left, splits = _censoredcoalsplits!(splits, distance(n[1]), init[id(n[1])])
#        rght, splits = _censoredcoalsplits!(splits, distance(n[2]), init[id(n[2])])
#        init[id(n)] = vcat(left, rght)
#    end
#    _, splits = _censoredcoalsplits!(splits, distance(tree), init[id(tree)])
#    return splits
#end
