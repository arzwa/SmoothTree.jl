# XXX: we'd better refactor a bit so that `Branches` always stores log-scale
# branch lengths, no we are doing to many log/exp interconversions. We hsould
# only exp them when sampling gene trees? (this also affects MSCModel and
# BranchModel...)
# Multivariate Gaussian model for EP ABC on fixed trees

# Define the natural parameter representation of a MvNormal
struct NaturalMvNormal{T<:Real,Cov<:AbstractMatrix,Mean<:AbstractVector}
    r::Mean
    Q::Cov
end

function NaturalMvNormal(r::AbstractVector{T}, Q::AbstractMatrix{T}) where T<:Real
    return NaturalMvNormal{T,typeof(Q),typeof(r)}(r, Q)
end

function tomoment(m::NaturalMvNormal)
    Qinv = m.Q^(-1)
    Σ = Symmetric(-0.5 * Qinv)
    μ = Σ * m.r
    return MvNormal(μ, Σ)
end

tonatural(m::MvNormal) = tonatural(m.μ, m.Σ)
function tonatural(μ, Σ)
    Σinv = Matrix(Σ^(-1))
    r = Vector(Σinv * μ)  # we don't want FillArrays etc...
    Q = -0.5 * Σinv
    return NaturalMvNormal(r, Q)
end

function logpartition(m::NaturalMvNormal)
    return -0.25 .* (m.r' * m.Q^(-1) * m.r) - 0.5 * log(det(-2*m.Q))
end

# not nice, should put a wrapper around NaturalMvNormal
logpdf(m::MvNormal, b::Branches) = logpdf(m, log.(b.xs))


# Now define what we need to do EP using the same methods as for the CCD
# models.
# Note that we use particles with as `S` field a vector of branch lengths.  We
# need to somehow be able to specify whether or not some branch should be
# represented (i.e. for branches with no information about branch length, often
# the tips).
# 1. Simulators
function simulate!(particles, model::NaturalMvNormal)
    mvn = tomoment(model)
    Threads.@threads for i=1:length(particles)
        simfun!(particles[i], mvn)
    end
end

# this is pretty straightforward now that we separated the splits from the
# distances in the `Branches` struct.
function simfun!(p, model::MvNormal)
    p.S = Branches(p.S.splits, exp.(rand(model)))
    p.w = logpdf(model, p.S)
end

# We specialize by converting to moment parameter once here...
function logweights(particles, model::NaturalMvNormal)
    w = zeros(length(particles))
    M = tomoment(model)
    Threads.@threads for i=1:length(particles)
        p = particles[i]
        w[i] = logpdf(M, p.S) - p.w
    end
    return w
end

# 2. Algebra
function mul!(m::NaturalMvNormal, a)
    m.r .*= a
    m.Q .*= a
    return m
end

function add!(x::NaturalMvNormal, y::NaturalMvNormal)
    x.r .+= y.r
    x.Q .+= y.Q
    return x
end

function sub!(x::NaturalMvNormal, y::NaturalMvNormal)
    x.r .-= y.r
    x.Q .-= y.Q
    return x
end

# 3. Match moments
function matchmoments_mvn(xs, ws)
    n = length(first(xs))
    μ = zeros(n)
    S = zeros(n,n)
    W = 0.
    for (ex, w) in zip(xs, ws)
        x = log.(ex)
        μ .+= w .* x
        S .+= w .* x*x'
        W += w
    end
    μ ./= W
    V = (S ./ W) .- (μ*μ')
    μ, Symmetric(V)
end

function matchmoments(branches, weights, cavity::NaturalMvNormal, args...)
    xs = getfield.(branches, :xs)
    return tonatural(matchmoments_mvn(xs, weights)...)
end

