
function ntrees(n)
    prod([i for i=1:2:2n-3])
end

function _p(n,k)
    n == k && return 1
    qn = 1/(2^(n-1) - 1)
    pnk = 0
    for i=1:(n-k)
        pnki = binomial(n, i)*_p(n-i,k)
        if n ÷ i == 2
            pnki /= 2
        end
        pnk += pnki
    end
    return qn*pnk
end
