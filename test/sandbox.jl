using SpecialFunctions, Plots

# Let us have a look at what the split *size* distributions look like 
f(β, n, i) = gamma(β+1+i)*gamma(β+1+n-i)/(gamma(i+1)*gamma(n-i+1))

function betasplitp(n, β)
    p = map(i->f(β, n, i), 1:n-1)
    p ./= sum(p)
    if β == -2.
        p[1] = p[end] = 0.5
    end
    return p
end

function _betasplitp(n, β)
    p = map(i->(2i==n ? 0.5 : 1.) * f(β, n, i), 1:n÷2)
    p ./= sum(p)
    return p
end

function uniformsplit(n)
    N = 2^(n-1) - 1
    p = 1/N
    map(i->binomial(n,i)*p/2, 1:n-1)
end

function _uniformsplit(n)
    N = 2^(n-1) - 1
    p = 1/N
    map(i->(2i==n ? 0.5 : 1.)*binomial(n,i)*p, 1:n÷2)
end

default(gridstyle=:dot, framestyle=:box)
n = 10
p = plot(size=(300,300))
for β=[-2, -1.5, -1., 0., 1., 10.]
    q = betasplitp(n, β)
    plot!(q, label=β)
end
plot!(uniformsplit(10), label="∞")
plot(p)

# for a (n,i) sized-split, there are (n pick i) different splits
# a split's probability can be computed by calculating the size
# probability and dividing by the number of possible such-sized
# splits.

# Now assume we have a Dirichlet prior distribution with α
# proportional to the β-split induced split probabilities. We next
# observe some splits and get a posterior 
#
# Clearly, observing a particular split of size j should not cause the
# other splits of size j to start differing in probability. This is a
# basic principle: all unrepresented splits of the same size have the
# same probability hence we must be able to store the probability
# distribution using the represented/observed splits and n-1 values
# for unrepresented splits (for a clade of size n)

struct DirichletBetaMBM{T}
    α::Vector{T}
    β::T
end

# get a distribution over splits from a distribution over split sizes
function DirichletBetaMBM(α, β, γ)
    n = SmoothTree.cladesize(γ)
    as = map(i->α * f(β, n, i), 1:n-1)
    αs = map(1:(γ+1)÷2) do δ
        x = SmoothTree.cladesize(δ)
        as[x]
    end
    DirichletBetaMBM(αs, β)
end

# consider a clade γ = 11111
# there are 2^4 - 1 = 15 splits
γ = UInt8(31)
d = DirichletBetaMBM(10., -1., γ)

dd = Dict(UInt8(3)=>2, UInt8(5)=>3)
bs = BetaSparseSplits(γ, dd, -1., 1.)
