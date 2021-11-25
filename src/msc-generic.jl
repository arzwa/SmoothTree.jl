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

# for with CCDs
const DefaultNode = Node{Int64,NewickData{Float64,String}}

function initdict(model::MSC, y::CCD)
    spleaves = Dict(name(n)=>id(n) for n in model.l)
    init = Dict(v => DefaultNode[] for v in values(spleaves))
    for (i,gene) in enumerate(y.leaves)
        species = spleaves[_spname(gene)]
        push!(init[species], Node(i, n=gene))
    end
    return init
end
   

 S = nw"((smo:1,(((gge:1,iov:1):1,(xtz:1,dzq:1):1):1,sgt:1):1):1,jvs:1);"
 julia> @btime SmoothTree.randtree(m);
  1.939 μs (66 allocations: 4.56 KiB)

