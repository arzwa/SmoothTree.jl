# root all trees identically
function rootall!(trees)
    tree = first(trees)
    leaf = name(first(getleaves(tree)))
    rootall!(trees, leaf)
end

function rootall!(trees, leaf)
    f = x->NewickTree.set_outgroup!(x, leaf)
    map(f, trees)
end
