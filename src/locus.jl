# by convention we assume genes are labeled "species__gene"
speciesname(x) = string(split(x, "__")[1])

"""
Stores the tree data for a single locus.
"""
struct Locus{D,T,V}
    data::D                       # a CCD or MBM
    lmap::BiMap{T,V}              # gene <=> clade index bimap
    init::Dict{T,Vector{T}}       # species tree taxa => extant gene tree clades map
    rooted::Bool
end

Base.length(x::Locus) = length(x.lmap)
Base.show(io::IO, x::Locus{D}) where D = write(io, "Locus $((length(x), D))")
randtree(locus::Locus{<:CCD}) = randtree(locus.data, locus.lmap)
randtree(locus::Locus{<:MomMBM}) = relabel!(randtree(locus.data), locus.lmap)

"""
    Locus(trees, spmap)

Get a conditional clade distribution (CCD) for the tree collection and put it
in the Locus data structure.
"""
function Locus(trees, spmap::BiMap{T}; rooted=true) where T
    lmap, init = getmaps(trees, spmap)
    ccd = CCD(trees, lmap, rooted=rooted)
    Locus(ccd, lmap, init, rooted)
end

"""
    Locus(trees, spmap, α, β)

Get a `β`-splitting posterior MBM with prior concentration `α` for the tree
collection and put it in the Locus data structure.
"""
function Locus(trees, spmap, α, β; rooted=true)
    lmap, init = getmaps(trees, spmap)
    ccd = CCD(trees, lmap, rooted=rooted)
    bsd = BetaSplitTree(β, length(lmap))
    mbm = MomMBM(ccd, bsd, α)
    Locus(mbm, lmap, init, rooted)
end

"""
    getmaps(trees, spmap)

Obtain a leaf <=> clade index map for a (collection of) gene tree(s) and
associate genes with species (to initialize coalescent simulations mainly).
"""
getmaps(trees::Vector, spmap) = getmaps(first(trees), spmap)
getmaps(trees::Dict, spmap) = getmaps(first(keys(trees)), spmap)

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



