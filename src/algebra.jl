# We need to implement the algebraic operations for models in natural
# parameterization for the EP algorithm.

# These functions are mutating in the first argument. 

# Algebra for NatBetaSplits
# =========================
function mul!(x::NatBetaSplits, a)
    for (γ, η) in x
        x[γ] *= a
    end
    x.η0 .*= a
    return x
end

function add!(x::NatBetaSplits, y::NatBetaSplits)
    @assert x.parent == y.parent
    splits = union(keys(x), keys(y))
    for δ in splits
        x[δ] += y[δ] 
        if !haskey(x, δ) 
            x.k[splitsize(x.γ, δ)] -= 1
        end
    end
    x.η0 .+= y.η0
    return x
end

function sub!(x::NatBetaSplits, y::NatBetaSplits)
    @assert x.parent == y.parent
    splits = union(keys(x), keys(y))
    for δ in splits
        x[δ] -= y[δ] 
        if !haskey(x, δ) 
            x.k[splitsize(x.γ, δ)] -= 1
        end
    end
    x.η0 .-= y.η0
    return x
end


# Non-mutating operations
# -----------------------
function Base.copy(x::NatBetaSplits)
    d = Dict(k=>v for (k,v) in x)
    NatBetaSplits(x.parent, x.refsplit, d, copy(x.n), copy(x.k), copy(x.η0)) 
end

Base.:*(a, x::NatBetaSplits) = x*a
function Base.:*(x::NatBetaSplits, a) 
    d = Dict(k=>a*v for (k,v) in x)
    NatBetaSplits(x.parent, x.refsplit, d, copy(x.n), copy(x.k), a .* copy(x.η0)) 
end

Base.:+(x::NatBetaSplits, y::NatBetaSplits) = add!(copy(x), y)
Base.:-(x::NatBetaSplits, y::NatBetaSplits) = sub!(copy(x), y)


# Algebra for CCD
# ===============
# This will only work if the underlying splits implement the required mul!,
# add! and sub! functions.

function mul!(x::CCD, a)
    for (k,v) in x
        mul!(v, a)
    end
    return x
end

function add!(x::CCD, y::CCD)
    for (γ, v) in y
        haskey(x, γ) ? add!(x[γ], v) : (x[γ] = v * 1.)
    end
    return x
end

function sub!(x::CCD, y::CCD)
    for (γ, v) in y
        haskey(x, γ) ? sub!(x[γ], v) : (x[γ] = v * -1.)
    end
    return x
end

Base.copy(x::CCD) = CCD(Dict(k => copy(v) for (k,v) in x), x.prior, x.root)
Base.:+(x::CCD, y::CCD) = add!(copy(x), y)
Base.:-(x::CCD, y::CCD) = sub!(copy(x), y)
Base.:*(x::CCD, a) = mul!(copy(x), a)
Base.:*(a, x::CCD) = mul!(copy(x), a)

