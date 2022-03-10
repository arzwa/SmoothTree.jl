using Turing, Optim
using Distributions, SmoothTree, NewickTree, Serialization

data = deserialize("docs/data/mints/mb1000.jls")
tre1 = readnw(readline("docs/data/mints/nepetoideae1.nw"))
taxa = name.(getleaves(tre1))
smap = clademap(taxa, UInt32)
root = UInt32(sum(keys(smap)))

X = map(data) do d
    SmoothTree.SplitCountsUnrooted(d, smap)
end

bsd = BetaSplitTree(-1.5, length(taxa))
model = CCD(SplitCounts(root), bsd, 1.)
logpdf(model, X[1])

@model betainfer(data, root, n) = begin
    β ~ Uniform(-2., 10.)
    bsd = BetaSplitTree(β, n)
    model = CCD(SplitCounts(root), bsd, 1.)
    for x in data
        Turing.@addlogprob!(logpdf(model, x))
    end
end

chain = sample(betainfer(X, root, length(taxa)), NUTS(), 200)

result = optimize(betainfer(X, root, length(taxa)), MLE())

mles = map(X) do x
    result = optimize(betainfer([x], root, length(taxa)), MLE())
    b = result.values[1]
end

# this does not work...
@model betainfer2(data, n) = begin
    β ~ Uniform(-2., 2.)
    for x in data
        bsd = BetaSplitTree(β, n)
        model = CCD(x, bsd, .1)
        Turing.@addlogprob!(logpdf(model, x))
    end
end

chain = sample(betainfer2(X, length(taxa)), NUTS(), 10)

# cross-validation?
# What is the best way to obtain the best (α,β) so that the CCD generalizes
# to unsampled trees in the best possible way?
