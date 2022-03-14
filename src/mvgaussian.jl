# Multivariate Gaussian model for EP ABC on fixed trees

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

function tonatural(m::MvNormal)
    Σinv = Matrix(m.Σ^(-1))
    r = Σinv * m.μ
    Q = -0.5 * Σinv
    return NaturalMvNormal(r, Q)
end

struct MvBranchModel{T,V}
end
