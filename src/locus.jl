abstract type AbstractLocus end

Base.length(x::AbstractLocus) = length(x.lmap)
randtree(locus::AbstractLocus) = randtree(locus.data, locus.lmap)
randtree(locus::AbstractLocus, n) = randtree(locus.data, locus.lmap, n)
maptree(locus::AbstractLocus) = maptree(locus.data, locus.lmap)
Base.show(io::IO, x::AbstractLocus) = write(io, "$(typeof(x)) $(length(x))")

# by convention we assume genes are labeled "species__gene"
speciesname(x) = string(split(x, "__")[1])

"""
Stores the tree data for a single locus.
"""
struct Locus{D,T,V} <: AbstractLocus
    data::D                       # a CCD
    lmap::BiMap{T,V}              # gene <=> clade index bimap
    init::Dict{T,Vector{T}}       # species tree taxa => extant gene tree clades map
    tag ::String                  # tag, usually file name
end

"""
    Locus(trees, spmap)

Get a conditional clade distribution (CCD) for the tree collection and put it
in the Locus data structure.
"""
function Locus(trees, spmap; prior=NoModel(), α=1., rooted=true, tag="")
    lmap, init = getmaps(trees, spmap)
    X = rooted ? SplitCounts(trees, lmap) : SplitCountsUnrooted(trees, lmap)
    ccd = CCD(X, prior, α)
    Locus(ccd, lmap, init, tag)
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

"""
    read_trprobs(fname; outgroup_prefix)

Read a MrBayes tree sample from the `trprobs` output file.
"""
function read_trprobs(fname; outgroup=nothing)
    ls = strip.(readlines(fname))
    start = findfirst(x->x=="translate", ls) + 1
    stop  = findfirst(x->startswith(x, "tree"), ls) - 1
    names = Dict(Pair(split(strip(y, [',']))...) for y in ls[start:stop])
    stop2 = findlast(x->startswith(x, "tree"), ls)
    map(stop+1:stop2) do i
        ws, ts = split(ls[i])[end-1:end]
        w = parse(Float64, strip(ws, [']']))
        t = relabel(readnw(string(ts)), names)
        if !isnothing(outgroup)
            l = getleaves(t)
            x = findfirst(x->startswith(name(x), outgroup), l)
            t = set_outgroup!(l[x])
        end
        t => w
    end |> Dict
end

getinit(l::Locus) = l.init
