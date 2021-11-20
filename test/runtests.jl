using Pkg; Pkg.activate(@__DIR__)
using SmoothTree, Test, NewickTree, Serialization

treesfile = joinpath(@__DIR__, "test/OG0006030.trees")

trees = readnw.(readlines(treesfile))
trees = SmoothTree.rootall!(trees)
ccd = CCD(trees)

@testset "Normalized" begin
    for (k,v) in ccd.cmap
        @test sum(values(ccd.smap[k])) ≈ v
    end
end

# MUL tree
trees = deserialize("test/treesummary.jls")

ccds = map(trees) do treeset
    xs = collect(treeset)
    counts = last.(xs)
    weights = counts / sum(counts)
    CCD(first.(xs), weights=weights)
end

X = SmoothTree.TreeData.(ccds)

S = nw"(((((dca,dca),dre),((dca,dca),dre)),((dca,dre),dca)),bvu);"
model = (nv=19, θ=exp.(randn(19)), S=S) 

SmoothTree.logpdf(model, X)

extree = collect(trees[3])[3][1]
X = SmoothTree.TreeData(CCD([extree]))
SmoothTree.logpdf(model, X)
