# do a NNI move on a tree, keeping the root in place though
nni_rooted(n) = nni_rooted!(deepcopy(n))
function nni_rooted!(n)
    #   ,-----A      ,-----B
    # --|p ,--B    --|  ,--A
    #   '--|n   ⟶    '--|
    #      '--C         '--C
    # we don't worry about branch lengths here
    p = parent(n)
    B = rand(children(n))
    A = sister(n)
    delete!(p, A)
    delete!(n, B)
    push!(n, A)
    push!(p, B)
    return getroot(p)
end

# A distribution like object for generating trees based on a true tree
# and a geometric/Poisson/... number of NNI rearrangements
struct RootedNNI{T,D}
    nodes::Vector{T}
    dist ::D
end

function RootedNNI(tree::T, d) where T<:Node
    f(n) = !(isroot(parent(n))) && !(isleaf(n))
    nodes = filter(f, prewalk(tree)[2:end])
    RootedNNI(nodes, d)
end

function randtree(d::RootedNNI)
    n = rand(d.dist)
    t = deepcopy(d.nodes)
    for i=1:n
        nni_rooted!(rand(t))
    end
    return getroot(t[1])
end

randtree(d::RootedNNI, n) = map(i->randtree(d), 1:n)
