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

# because why not
function Base.show(io::IO, m::BiMap{T,V}) where {T,V} 
    write(io, "$(typeof(m)) with $(length(m.m1)) entries:")
    for (k,v) in m.m1
        write(io, "\n  $(repr(k)) <=> $(repr(v))")
    end
end

# a taxonmap
taxonmap(pair::Pair, T=UInt16) = taxonmap(first(pair), T)
taxonmap(tree, T=UInt16) = taxonmap(name.(getleaves(tree)), T)

function taxonmap(l::Vector{String}, T=UInt16)
    d = Dict(T(2^i)=>l[i+1] for i=0:length(l)-1)
    return BiMap(d)
end

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

