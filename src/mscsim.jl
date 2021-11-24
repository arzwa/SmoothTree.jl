struct MSC{V}
    S::V
    o::Vector{V}
    l::Vector{V}
end

MSC(tree) = MSC(tree, postwalk(tree), getleaves(tree))

function setdistance!(S, θ::Vector)
    for (i,n) in enumerate(postwalk(S))
        isroot(n) && return
        n.data.distance = θ[i]
    end
end

function setdistance!(S, θ::Number)
    for (i,n) in enumerate(postwalk(S))
        isroot(n) && return
        n.data.distance = θ
    end
end

function randtree(model::MSC) 
    init = Dict(id(n)=>[Node(i, n=name(n))] for (i,n) in enumerate(model.l))
    randtree(model, init)
end

function randtree(model::MSC, n::Int) 
    init = Dict(id(n)=>[Node(i, n=name(n))] for (i,n) in enumerate(model.l))
    randtree(model, init, n)
end

randtree(model::MSC, init::Dict, n::Int) = map(i->randtree(model, copy(init)), 1:n)

function randtree(model::MSC, init::Dict)
    i = length(init) + 1
    for snode in model.o[1:end-1]
        i = censored_coal!(init, snode, i)
    end
    finish_coal(init[id(model.o[end])], i)
end

function finish_coal(nodes, i)
    while length(nodes) > 1
        shuffle!(nodes)
        a = pop!(nodes)
        b = pop!(nodes)
        c = Node(i)
        push!(c, a, b)
        push!(nodes, c)
        i += 1
    end
    pop!(nodes)
end

function censored_coal!(uncoal, snode, i)
    potential = uncoal[id(snode)]
    n = length(potential)
    t = distance(snode)
    t -= randexp() * 2 / (n*(n-1))
    while t > 0
        shuffle!(potential)
        c = Node(i)
        push!(c, potential[end-1], potential[end])
        potential = [potential[1:end-2] ; c]
        n = length(potential)
        t -= randexp() * 2 / (n*(n-1))
        i += 1
    end
    p = id(parent(snode))
    uncoal[p] = haskey(uncoal, p) ? vcat(uncoal[p], potential) : potential
    return i
end

# Would be cool to simulate a CCD directly

# S = ((smo:10.0,(((gge:9.0,iov:6.0):7.0,(xtz:11.0,dzq:9.0):7.0):23.0,sgt:8.0):2.0):7.0,jvs:7.0);
# julia> @btime SmoothTree.mscsim(S)
#  2.645 μs (80 allocations: 5.74 KiB)
#  ((smo,(sgt,((dzq,xtz),(gge,iov)))),jvs);

