# Different tree representations and interconversions...
# 1. NewickTree trees
# 2. Split sets (cladograms)
# 3. Branch sets (phylogenies)
# Why not work with NewickTree trees everywhere? Recursive data structure vs.
# array based?

# NewickTree tree, default type
const DefaultNode{T} = Node{T,NewickData{Float64,String}}
const SimpleNode{T} = Node{T,Float64}
NewickTree.distance(n::SimpleNode) = n.data
NewickTree.name(n::SimpleNode) = n.id

# we extend NewickTree here, because we deal a lot with nameless trees
NewickTree.name(n::DefaultNode) = n.data.name == "" ? string(id(n)) : n.data.name

"""
    Splits{T}

A vector representation of a tree topology (cladogram)
"""
const Splits{T} = Vector{Tuple{T,T}}

function getsplits(n::DefaultNode{T}, m::AbstractDict) where T
    splits = Splits{T}()
    _getsplits(splits, n, m)
    return splits
end

function _getsplits(splits, n, m)
    isleaf(n) && return m[name(n)]
    a = _getsplits(splits, n[1], m)
    b = _getsplits(splits, n[2], m)
    push!(splits, (a + b, min(a,b)))
    return a + b
end

"""
    Branches{T}

A vector representation of a phylogenetic tree (with `Float64` branch lengths).
"""
const Branches{T} = Vector{Tuple{T,T,Float64}}

function getbranches(n::DefaultNode{T}, m::AbstractDict) where {T}
    branches = Branches{T}()
    _getbranches(branches, n, m)
    return reverse(branches)  # return in preorder
end

function _getbranches(branches, n, m)
    isleaf(n) && return m[name(n)], distance(n)
    a, da = _getbranches(branches, n[1], m)
    b, db = _getbranches(branches, n[2], m)
    push!(branches, (a + b, a, da))
    push!(branches, (a + b, b, db))
    return a + b, distance(n)
end

function getbranchdict(n, m)
    Dict((x[1],x[2])=>x[3] for x in getbranches(n, m))
end

# obtain a tree from a split set
function gettree(splits::Splits{T}, names) where T
    nodes = Dict{T,DefaultNode{T}}()
    for (γ, δ) in splits
        p, l, r = map(c->_getnode!(nodes, names, c), [γ, δ, γ-δ])
        push!(p, l, r)   
    end
    return getroot(nodes[splits[end][1]])
end

function _getnode!(nodes, names, n)
    isleafclade(n) && return Node(n, n=names[n])
    haskey(nodes, n) && return nodes[n]
    nodes[n] = Node(n)
    return nodes[n]
end

function gettree(branches::Branches{T}, names) where T
    nodes = Dict{T,DefaultNode{T}}()
    for (γ, δ, d) in branches
        c = _getnode!(nodes, names, δ)
        c.data.distance = d
        p = _getnode!(nodes, names, γ)
        push!(p, c)   
    end
    return getroot(nodes[branches[1][1]])
end

