struct MSC1{V}
    S::V
    o::Vector{V}
    l::Vector{V}
end

MSC1(tree) = MSC1(tree, postwalk(tree), getleaves(tree))

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

function randtree(model::MSC1) 
    init = Dict(id(n)=>[Node(i, n=name(n))] for (i,n) in enumerate(model.l))
    randtree(model, init)
end

function randtree(model::MSC1, n::Int) 
    init = Dict(id(n)=>[Node(i, n=name(n))] for (i,n) in enumerate(model.l))
    randtree(model, init, n)
end

randtree(model::MSC1, init::Dict, n::Int) = map(i->randtree(model, copy(init)), 1:n)

function randtree(model::MSC1, init::Dict)
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

