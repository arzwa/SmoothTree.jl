using Pkg; Pkg.activate(@__DIR__)
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

chain = sample(betainfer(X, root, length(taxa)), NUTS(), 500)

#Chains MCMC chain (500×13×1 Array{Float64, 3}):
#
#Iterations        = 1:1:500
#Number of chains  = 1
#Samples per chain = 500
#parameters        = β
#internals         = acceptance_rate, hamiltonian_energy, hamiltonian_energy_error, is_accept, log_density, lp, max_hamiltonian_energy_error, n_steps, nom_step_size, numerical_error, step_size, tree_depth
#
#Summary Statistics
#  parameters      mean       std   naive_se      mcse        ess      rhat
#      Symbol   Float64   Float64    Float64   Float64    Float64   Float64
#
#           β   -1.1496    0.0006     0.0000    0.0000   330.9248    0.9996
#
#Quantiles
#  parameters      2.5%     25.0%     50.0%     75.0%     97.5%
#      Symbol   Float64   Float64   Float64   Float64   Float64
#
#           β   -1.1506   -1.1500   -1.1496   -1.1492   -1.1485
#

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
