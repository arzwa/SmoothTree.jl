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

clademap(tree::Node, args...) = clademap(name.(getleaves(tree)), args...)

function clademap(taxa::Vector{String}, ::Type{T}=cladetype(taxa)) where T
    l = sort(taxa)
    d = Dict(T(2^i)=>l[i+1] for i=0:length(l)-1)
    return BiMap(d)
end

function cladetype(taxa)
    n = length(taxa)
    @assert n <= 64
    T = n <= 8 ? UInt8 : n <= 16 ? UInt16 : n <= 32 ? UInt32 : UInt64
    return T
end

function getclade(m::BiMap{T,String}, clade::Vector{String}) where T  
    return T(sum([m[x] for x in clade]))
end
