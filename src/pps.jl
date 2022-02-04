# Posterior predictive simulation
function merge_pmaps(xs)
    ks = union(keys.(xs)...)
    d = Dict(k=>zeros(length(xs)) for k in ks)
    for i in 1:length(xs)
        for (k,v) in xs[i]
            d[k][i] = v
        end
    end
    return d
end

function postpredsim(model, tmap, n, N)
    trees = Dict()
    pps = map(1:N) do rep
        S = randtree(model)
        M = MSC(S, Dict(id(x)=>[id(x)] for x in getleaves(S)))
        ts = randtree(M, tmap, n)
        pps = proportionmap(hash.(ts))
    end
    merge_pmaps(pps)
end
