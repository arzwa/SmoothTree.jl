
# Blum & Francois models
ntrees(n) = prod([i for i=1:2:2n-3])
pda(n,i) = (binomial(n, i) * ntrees(n-i) * ntrees(i) // ntrees(n)) // 2
harmonic(n) = sum([1//i for i=1:n])
ab(n,i) = (n // (i * (n-i))) // (2*harmonic(n-1))

# uniform splits
function unif(n,i) 
    x = binomial(n,i)//(2^n-2)
    n == 2i ? x/2 : x
end

function _p(n,k)
    n == k && return 1
    f = 0
    for i=0:(n-k-1)
        f += _p(k+i,k) * unif(k+i,k)
    end
    return f
end
