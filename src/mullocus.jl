# A Locus struct for MUL trees
# ============================
# We assume the genes come from distinct subgenomes (at most a single gene is
# sampled per subgenome).

struct MulLocus{D,T,V} <: AbstractLocus
    data::D
    lmap::BiMap{T,V}
    sg2sp::Dict{T,V}
    sp2gene::Dict{V,Vector{T}}
    tag::String
    # `spprobs`: probability vectors for each gene -> species assignment
end

function MulLocus(trees, spmap; prior=NoModel(), α=1., rooted=true, tag="")
    lmap, sg2sp, sp2gene = getmulmaps(trees, spmap)
    X = rooted ? SplitCounts(trees, lmap) : SplitCountsUnrooted(trees, lmap)
    ccd = CCD(X, prior, α)
    MulLocus(ccd, lmap, sg2sp, sp2gene, tag)
end

getmulmaps(trees::Vector, spmap) = getmulmaps(first(trees), spmap)
getmulmaps(trees::Dict, spmap) = getmulmaps(first(keys(trees)), spmap)

function getmulmaps(tree::Node, spmap::BiMap{T,V}) where {T,V}
    genes = sort(name.(getleaves(tree)))
    lmap = clademap(genes, T)
    sg2sp = Dict(k=>speciesname(v) for (k,v) in spmap)
    sp2gene = Dict(v=>T[] for (k,v) in sg2sp)
    for (g, n) in lmap
        push!(sp2gene[speciesname(n)], g)
    end
    return lmap, sg2sp, sp2gene
end

# not so nice
function randinit(l::MulLocus{D,T}) where {D,T}
    d = Dict{T,Vector{T}}()
    genes = Dict(k=>shuffle(v) for (k,v) in l.sp2gene)
    sgs = shuffle(collect(keys(l.sg2sp)))
    for k in sgs
        v = l.sg2sp[k]
        d[k] = isempty(genes[v]) ? T[] : [pop!(genes[v])]
    end
    return d
end

getinit(l::MulLocus) = randinit(l)
