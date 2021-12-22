"""
    BiMap{T,V}

Bijective map for distinct types, so that `getindex` dispatches on
type.
"""
struct BiMap{T,V}
    m1::Dict{T,V}
    m2::Dict{V,T}
    function BiMap(d::Dict{T,V}) where {T,V}
        @assert T != V  "BiMap only defined for distinct types"
        new{T,V}(d, Dict(v=>k for (k,v) in d))
    end
end

Base.getindex(m::BiMap{T,V}, k::T) where {T,V} = m.m1[k]
Base.getindex(m::BiMap{T,V}, k::V) where {T,V} = m.m2[k]
Base.haskey(m::BiMap{T,V}, k::T) where {T,V} = haskey(m.m1, k)
Base.haskey(m::BiMap{T,V}, k::V) where {T,V} = haskey(m.m2, k)

# because why not
function Base.show(io::IO, m::BiMap{T,V}) where {T,V} 
    write(io, "$(typeof(m)) with $(length(m.m1)) entries:")
    for (k,v) in m.m1
        write(io, "\n  $(repr(k)) <=> $(repr(v))")
    end
end

# a taxonmap
taxonmap(tree, T=UInt16) = taxonmap(name.(getleaves(tree)), T)

function taxonmap(l::Vector{String}, T=UInt16)
    d = Dict(T(2^i)=>l[i+1] for i=0:length(l)-1)
    return BiMap(d)
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
