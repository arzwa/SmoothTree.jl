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

# relabel a tree based on an id<=>string map
relabel(trees::Vector, m) = relabel.(trees, Ref(m))
relabel(tree, m) = relabel!(deepcopy(tree), m)
function relabel!(tree, m)
    for n in getleaves(tree)
        n.data.name = m[id(n)]
    end
    return tree
end

# root all trees identically
function rootall(trees)
    tree = first(trees)
    leaf = name(first(getleaves(tree)))
    rootall(trees, leaf)
end

function rootall(trees, leaf)
    f = x->getroot(NewickTree.set_outgroup(x, leaf))
    map(f, trees)
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

# decompose a tree in a set of clades
function getclades(tree)
    clades = Vector{String}[]
    function walk(n)
        clade = isleaf(n) ? [name(n)] : [walk(n[1]) ; walk(n[2])]
        sort!(clade)
        push!(clades, clade)
        return clade
    end
    walk(tree)
    sort!(clades)
    return clades
end

# test equality of cladograms, allows for countmaps/proportionmaps
Base.hash(tree::Node) = hash(getclades(tree))
Base.isequal(t1::Node, t2::Node) = hash(t1) == hash(t2)

# compatibility
⊂(δ::T, γ::T) where T<:Integer = 
    δ < γ && cladesize(γ) == cladesize(δ) + cladesize(γ-δ)

