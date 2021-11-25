using Pkg; Pkg.activate(@__DIR__)
using SmoothTree, Test, NewickTree, Serialization, StatsBase, Distributions

@testset "SmoothTree tests" begin
    treesfile = joinpath(@__DIR__, "test/OG0006030.trees")
    trees = readnw.(readlines(treesfile))
    trees = SmoothTree.rootall!(trees)
    ccd = CCD(trees)

    @testset "Proper normalization?" begin
        for (k,v) in ccd.cmap
            SmoothTree.isleafclade(k) && continue
            @test sum(values(ccd.smap[k])) ≈ v
        end
    end
end
