# CCD based approach for inferring a MUL tree from a bunch of gene
# trees with missing genes. This is more or less a simplified version
# of ALE, but with the ultimate goal to infer a species tree, more or
# less like a supertree method?

struct TreeData{T,V}
    ccd::CCD{T,V}
    clades::Vector{T}
    species::Vector{Vector{String}}
    cladeindex::Dict{T,Int64}
end

function TreeData(X::CCD)
    clades = sort(collect(keys(X.cmap)))
    species = map(clade->_spname.(getclade(X, clade)), clades)
    cladeindex = Dict(c=>i for (i,c) in enumerate(clades))
    TreeData(X, clades, species, cladeindex)
end

Base.show(io::IO, X::TreeData) = write(io, "TreeData($(X.ccd))")

logpdf(model, X::Vector{<:TreeData}) = mapreduce(x->logpdf(model, x), +, X)

function logpdf(model, X::TreeData)
    @unpack nv, θ, S = model  
    @unpack ccd, clades, species, cladeindex = X
    # number of vertices, params and species tree
    # θ is a vetor of branch lengths (μt products, with μ the loss
    # rate and t the timespan of the branch). 
    # We assume they are provided in the order corresponding to the
    # postorder on S.
    L = fill(-Inf, length(clades), nv)
    for node in postwalk(S)
        u = id(node)
        maxclade = name.(getleaves(node))
        for (i,clade) in enumerate(clades)  
            s = join(species[i], " ")
            # the order doesn't matter here, but we should have one
            # if incompatible, skip
            species[i] ⊈ maxclade && continue  
            if isleafclade(clade) && isleaf(node)
                L[i,u] = 0.
            # note that in the MUL tree case, we may have a non leaf
            # clade which is compatible with a leaf node, so we need
            # to check !isleaf 
            elseif !isleaf(node) 
                v = id(node[1])
                w = id(node[2])
                ℓ = -Inf 
                z = log(ccd.cmap[clade])
                dv = log1mexp(-θ[v]) 
                dw = log1mexp(-θ[w])
                # represented split
                if !isleafclade(clade)
                    for (γ, p) in ccd.smap[clade]
                        ccp = log(p) - z
                        j = cladeindex[γ]
                        k = cladeindex[clade-γ]
                        s1 = join(species[j], " ")
                        s2 = join(species[k], " ")
                        l = ccp + logaddexp(L[j,v]+L[k,w], L[k,v]+L[j,w])
                        ℓ = logaddexp(ℓ, l + dw + dv)
                    end
                end
                # loss event
                ℓ = logaddexp(ℓ, L[i,v] + dv -θ[w])
                ℓ = logaddexp(ℓ, L[i,w] + dw -θ[v])
                L[i,u] = logaddexp(L[i,u], ℓ)
            end
        end
    end
    ℓ = L[end,id(S)]  # log likelihood
    return L
    return ℓ
end


