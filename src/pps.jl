
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

# note, this is simulating data sets similar to the input data, but not
# sampling gene trees for the actual input data from the posterior.
function postpredsim(model::MSCModel{T}, data, n) where T
    results = Vector{Dict{DefaultNode{T}, Float64}}(undef, n)
    Threads.@threads for i=1:n
        S = randbranches(model)
        G = map(x->gettree(randsplits(S, x.init), x.lmap), data)
        results[i] = proportionmap(G)
    end
    return merge_pmaps(results)
end
