
# find for each node in S all clades that can be seen at the ancestral
# edge of the associated branch
function findclades(S, X::TreeData{T}) where T
    order = postwalk(S)
    clades = [T[] for i=1:length(order)]
    for node in order
        species = name.(getleaves(node))
        for (i, clade) in enumerate(X.clades)
            X.species[i] ⊆ species && push!(clades[id(node)], clade)
        end
    end
    return clades
end
