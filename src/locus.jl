# get the lmap out of the CCD. Why? e.g. when simulating species trees in the
# EP-ABC, we should be able to get a CCD without requiring some species name
# map. We should also be able to get a CCD just from splits, irrespective of
# what these splits are named etc.
# Locus will be a bundling of these things then...

speciesname(x) = string(split(x, "_")[1])

"""
Stores the tree data for a single locus.
"""
struct Locus{D,T,V}
    data::D                       # a CCD or MBM
    lmap::BiMap{T,V}              # gene <=> clade index bimap
    init::Dict{T,Vector{T}}       # species tree taxa => extant gene tree clades map
end

Base.length(x::Locus) = length(x.lmap)
Base.show(io::IO, x::Locus{D}) where D = write(io, "Locus $((length(x), D))")

"""
    Locus(trees, spmap)

Get a conditional clade distribution (CCD) for the tree collection and put it
in the Locus data structure.
"""
function Locus(trees, spmap)
    lmap, init = getmaps(trees, spmap)
    ccd = CCD(trees, lmap)
    Locus(ccd, lmap, init)
end

"""
    Locus(trees, spmap, α, β)

Get a `β`-splitting posterior MBM with prior concentration `α` for the tree
collection and put it in the Locus data structure.
"""
function Locus(trees, spmap, α, β)
    lmap, init = getmaps(trees, spmap)
    ccd = CCD(trees, lmap)
    bsd = BetaSplitTree(β, length(lmap))
    mbm = MomMBM(ccd, bsd, α)
    Locus(mbm, lmap, init)
end

"""
    getmaps(trees, spmap)

Obtain a leaf <=> clade index map for a (collection of) gene tree(s) and
associate genes with species (to initialize coalescent simulations mainly).
"""
getmaps(trees::Vector, spmap) = getmaps(first(trees), spmap)

function getmaps(tree::Node, spmap::BiMap{T,V}) where {T,V}
    genes = sort(name.(getleaves(tree)))
    lmap = clademap(genes, T)
    init = Dict(x=>T[] for x in keys(spmap))
    for gene in genes
        species = speciesname(gene)
        push!(init[spmap[species]], lmap[gene])
    end
    return lmap, init
end

function clademap(l::Vector{String}, ::Type{T}) where T
    d = Dict(T(2^i)=>l[i+1] for i=0:length(l)-1)
    return BiMap(d)
end



