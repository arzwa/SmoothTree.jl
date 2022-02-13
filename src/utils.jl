# the number of possible splits of a set of size n
nsplits(n) = 2^(n-1) - 1

# number of rooted trees with n taxa
ntrees(n) = prod([2i-3 for i=3:n])

# ranking
ranking(xs) = sort(collect(proportionmap(xs)), by=last, rev=true)

# remove branch lengths
topologize(tree) = topologize!(deepcopy(tree))
function topologize!(tree)
    for n in postwalk(tree)
        n.data.distance = NaN
        n.data.support = NaN
    end
    return tree
end

"""
    BiMap{T,V}

Bijective map for distinct types, so that `getindex` dispatches on
type.
"""
struct BiMap{T,V} <: AbstractDict{T,V}
    m1::Dict{T,V}
    m2::Dict{V,T}
    function BiMap(d::Dict{T,V}) where {T,V}
        @assert T != V  "BiMap only defined for distinct types"
        new{T,V}(d, Dict(v=>k for (k,v) in d))
    end
end

Base.length(m::BiMap) = length(m.m1)
Base.getindex(m::BiMap{T,V}, k::T) where {T,V} = m.m1[k]
Base.getindex(m::BiMap{T,V}, k::V) where {T,V} = m.m2[k]
Base.haskey(m::BiMap{T,V}, k::T) where {T,V} = haskey(m.m1, k)
Base.haskey(m::BiMap{T,V}, k::V) where {T,V} = haskey(m.m2, k)
Base.iterate(m::BiMap) = Base.iterate(m.m1)
Base.iterate(m::BiMap, i) = Base.iterate(m.m1, i)

function Base.show(io::IO, m::BiMap{T,V}) where {T,V} 
    write(io, "$(typeof(m)) with $(length(m.m1)) entries:")
    for (k,v) in m.m1
        write(io, "\n  $(repr(k)) <=> $(repr(v))")
    end
end

# a clademap
clademap(tree::Node{T}) where T = clademap(name.(getleaves(tree)), T)
clademap(tree, T) = clademap(name.(getleaves(tree)), T)
clademap(trees::AbstractDict) = clademap(collect(keys(trees)))
clademap(trees::Vector{<:Node{T}}) where T = clademap(trees, T)

function clademap(trees::Vector{<:Node}, T)
    clademap(unique(mapreduce(x->name.(getleaves(x)), vcat, trees)), T)
end

function clademap(l::Vector{String}, T=UInt16)
    d = Dict(T(2^i)=>l[i+1] for i=0:length(l)-1)
    return BiMap(d)
end

getclade(m::BiMap{T}, clade::Vector{String}) where T = 
    T(sum([m[x] for x in clade]))

# relabel a tree based on an id<=>string map
relabel(tree, m) = relabel!(deepcopy(tree), m)
function relabel!(tree, m)
    for n in getleaves(tree)
        n.data.name = m[id(n)]
    end
    return tree
end

# root all trees identically
function rootall!(trees)
    tree = first(trees)
    leaf = name(first(getleaves(tree)))
    rootall!(trees, leaf)
end

function rootall!(trees, leaf)
    f = x->getroot(NewickTree.set_outgroup!(x, leaf))
    map(f, trees)
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
# about species tree branch lengths for the branches coming from the
# root
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

function process_data(dpath, outgroup)
    fnames = readdir(dpath, join=true)
    map(fnames) do fpath
        @info fpath
        ts = readnw.(readlines(fpath))
        topologize!.(ts)
        ts = SmoothTree.rootall!(ts, outgroup)
        countmap(ts)
    end
end

# some IO for tree collections...
writetrees(path, cmaps::Vector) = map(x->writetrees(path, x), cmaps)
function writetrees(path, cmap::AbstractDict)
    open(path, "a") do f
        for (k,v) in sort(collect(cmap), by=last, rev=true)
            write(f, "$v\t$(nwstr(k))\n")
        end
        write(f, "***\n")
    end
end

function readtrees(path)
    content = readchomp(open(path, "r"))
    content = split(content, "***\n")
    map(content) do x
        xs = filter(x->length(x) == 2, map(x->split(x, "\t"), split(x, "\n")))
        Dict(readnw(string(x[2]))=>parse(Int, x[1]) for x in xs)
    end
end

# Tree comparison etc.
getcladesbits(tree, T=UInt16) = getcladesbits(tree, clademap(tree))

# get clades as bitstrings
function getcladesbits(tree, m::BiMap{T,V}) where {T,V}
    clades = T[]
    function walk(n)
        clade = isleaf(n) ? m[name(n)] : walk(n[1]) + walk(n[2])
        push!(clades, clade)
        return clade
    end
    walk(tree)
    return clades
end

# XXX the above does not work as a hashing function! isomorphic trees
# (same topology, different labels) will have the same clade set!
# This does lead to the following handy function
function isisomorphic(t1, t2, tmap)
    h1 = hash(sort(getcladesbits(t1, tmap)))
    h2 = hash(sort(getcladesbits(t2, tmap)))
    h1 == h2
end

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

# compatibility
⊂(δ::T, γ::T) where T<:Integer = 
    δ < γ && cladesize(γ) == cladesize(δ) + cladesize(γ-δ)

#function ⊂(δ::T, γ::T) where T<:Integer 
#    δ > γ && return false
#    a = digits(γ, base=2)
#    b = digits(δ, base=2)
#    all(a[1:length(b)] .- b .>= 0)
#end


