# A BMP or Branching Markov process is a distribution on tree
# topologies for a fixed leaf set L defined by a set of conditional
# split distributions, i.e. for each possible clade of L, we specify a
# probability distribution over splits. In general, we are interested
# in sparsely represented BMP distributions, where we do not need to
# consider the full split distributions *explicitly*.

# A Dirichlet-BMP is a conjugate prior for a BMP, which instead of
# conditional split distributions consists of a set of Dirichlet
# distributions, realizations of which are then the conditional split
# distributions.

# A BMP is the same as a CCD, but I reserve CCD to refer to an
# estimator of a CCD based on a sample of trees (i.e. computing the
# CCD using Larget's algorithm, the maximum entropy distribution on
# tree topologies constrained by observed marginal split frequencies
# (cfr. Szollosi)).

# A potentially unnormalized BMP in natural parameter space: this
# means we have to maintain for each clade a distribution over splits
# and normalization factor. It is probably easiest to maintain these
# in separate fields (not using a special key in the dict)
# We mainly want to be able to sample from the unnormalized BMP and to
# efficiently construct the cavity distribution and convex
# combinations of BMPs.
struct BMP{T,V}
    smap::Dict{T,Dict{T,V}}
    zmap::Dict{T,Tuple{Float64,Float64,Int64}}   # Z, p0 and k
    root::T
end

# Uniform BMP model
UniformBMP(n, T=UInt16) = BMP(Dict{T,Dict{T,Float64}}(), rootclade(n, T)) 

# mean of a Dirichlet-BMP is a BMP
function Distributions.mean(x::CCD{T}) where T
    zmap = Dict{T,Tuple{Float64,Float64,Int64}}()
    smap = Dict{T,Dict{T,Float64}}()
    map(collect(x.smap)) do (γ, d)
        dd, zs = _splitfun(γ, d, x.α)
        zmap[γ] = zs
        smap[γ] = dd 
    end
    return BMP(smap, zmap, x.root)
end

function _splitfun(γ, d, α)
    n = _ns(cladesize(γ))
    k = length(d)
    N = sum(values(d))
    Z = log(α * n + N)
    # represented splits
    dd = Dict(c=>log(α + v) for (c, v) in d) 
    # unrepresented: 
    p = n - k == 0 ? 0 : log(α) - Z
    return dd, (Z, p, k)
end

# Base extensions
Base.keys(model::BMP) = keys(model.smap)
Base.haskey(model::BMP{T}, γ::T) where T = haskey(model.smap, γ) 
Base.haskey(model::BMP{T}, γ::T, δ::T) where T = 
    haskey(model.smap, γ) && haskey(model.smap[γ], δ)
Base.getindex(model::BMP{T}, γ::T) where T = model.smap[γ]

# deal with sparsely represented BMPs
nrepresented(model::BMP, γ) = haskey(model, γ) ? length(model[γ]) : 0
#represented_mass(model::BMP, γ) = haskey(model, γ) ? sum(values(model[γ])) : 0

# draw a random tree from a BMP, node id's are clade indices
randtree(model::BMP) = _randwalk(Node(model.root), model)

# recursion for randtree
function _randwalk(node, model)
    clade = id(node)
    isleafclade(clade) && return
    splt = randsplit(model, clade)
    n1 = Node(splt, node)
    n2 = Node(clade - splt, node)
    _randwalk(n1, model)
    _randwalk(n2, model)
    return node
end

# generate a random split for a clade according to a sparsely
# represented BMP
function randsplit(model::BMP, clade)
    n = cladesize(clade)
    !haskey(model, clade) && return randsplit(clade)
    splitps = collect(model[clade])
    weights = last.(splitps) .- model.zmap[clade][1]
    r = logsumexp(weights)
    #r = represented_mass(model, clade)
    if rand() < r
        i = sample(1:length(weights), Weights(weights))
        splt = first(splitps[i])
    else
        splt = randsplit(clade)
    end
    return splt
end

# not sure if correct, and also not sure if numerically stable
# enough...
function cavity(model::BMP{T,V}, factor::BMP{T,V}) where {T,V}
    newd = Dict{T,Dict{T,V}}()
    clades = collect(union(keys(model), keys(factor)))
    for clade in clades
        newd[clade] = Dict{T,V}()
        n = cladesize(clade)
        r1 = represented_mass(model, clade)
        k1 = nrepresented(model, clade)
        r2 = represented_mass(factor, clade)
        k2 = nrepresented(factor, clade)
        p01 = (1-r1) / (n-k1)
        p02 = (1-r2) / (n-k2)
        p0 = p01/p02
        for (sk, pk) in model[clade]
            if haskey(factor, clade, sk)
                newd[clade][sk] = pk/factor[clade][sk]
            else
                newd[clade][sk] = pk/p02
            end
        end
        if haskey(factor, clade)
            for (sk, pk) in factor[clade]
                haskey(newd[clade], sk) && continue  # already seen
                newd[clade][sk] = p01/pk
            end
        end
        # renormalize
        s1 = sum(values(newd[clade]))
        s2 = p0*(n - length(newd[clade]))
        Z = s1 + s2 
        for (sk, pk) in newd[clade]
            newd[clade][sk] /= Z
        end
    end
    return BMP(newd, model.root)
end

function fitbmp(trees, taxon_map::Dict{String,T}) where T
    smap = Dict{T,Dict{T,Float64}}()
    w = 1/length(trees)
    for tree in trees 
        _addtree!(smap, tree, w, taxon_map)
    end
    # renormalization
    for (k, v) in smap
        Z = sum(values(v))
        for (s, p) in v
            v[s] = p/Z
        end
    end
    return BMP(smap, maximum(keys(smap)))
end

function _addtree!(smap, node, w, m)
    isleaf(node) && return m[name(node)]
    left = _addtree!(smap, node[1], w, m)
    rght = _addtree!(smap, node[2], w, m)
    clade = left + rght
    x = left < rght ? left : rght
    if !haskey(smap, clade)
        smap[clade] = Dict(x=>0.)
    end
    if !haskey(smap[clade], x)
        smap[clade][x] = 0.
    end
    smap[clade][x] += w
    return clade
end

taxon_map(leaves, T=UInt16) = Dict(T(2^i)=>leaves[i+1] for i=0:length(leaves)-1)
