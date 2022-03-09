using Turing
using Distributions, SmoothTree, NewickTree, Serialization

data = deserialize("docs/data/mints/mb1000.jls")
tre1 = readnw(readline("docs/data/mints/nepetoideae1.nw"))
taxa = name.(getleaves(tre1))
smap = clademap(taxa, UInt32)
root = UInt32(sum(keys(smap)))

data = map(data) do d
    Dict(k=>v/sum(values(d)) for (k,v) in d)
end

# we can compare using ML gene trees against acknowledging uncertainty
data1 = Locus.(data, Ref(smap), 1e-6, -1.5)
data2 = [SmoothTree.getsplits(data[i], data1[i].lmap) for i=1:length(data)]

function loglik(root, data, β)
    n = SmoothTree.cladesize(root)
    mapreduce(i->logpdf(NatMBM(root, BetaSplitTree(β, n)), data[i]), +, 1:length(data))
end

# We can easily find posterior distribution for beta using above likelihood.
# Somehow it sounds reasonable that the posterior mean beta should be the ideal
# value for smoothing the tree empirical CCD, but how does this make sense
# formally?

@model betainfer(data, root) = begin
    β ~ Uniform(-2., 2.)
    Turing.@addlogprob!(loglik(root, data, β))
end

chain = sample(betainfer(data2, root), NUTS(), 500)

#For the mints data set:
#Summary Statistics
#  parameters      mean       std   naive_se      mcse        ess    ⋯
#      Symbol   Float64   Float64    Float64   Float64    Float64    ⋯
#
#           β   -1.3009    0.0145     0.0006    0.0013   110.7126    ⋯
#                                                     1 column omitted
#
#Quantiles
#  parameters      2.5%     25.0%     50.0%     75.0%     97.5%
#      Symbol   Float64   Float64   Float64   Float64   Float64
#
#           β   -1.3288   -1.3098   -1.3011   -1.2913   -1.2720
#
