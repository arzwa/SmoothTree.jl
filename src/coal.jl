
# probability that u lineages coalesce to v lineages over time t
function pcoal(u, v, t)
    p = 0.
    for k=v:u
        x = exp(-k*(k-1)*t/2) 
        x *= (2k-1)*(-1)^(k-v)
        x /= (factorial(v) * factorial(k-v) * (v+k-1))
        x *= prod([(v+y)*(u-y)/(u+y) for y=0:(k-1)])
        p += x
    end
    return p
end


