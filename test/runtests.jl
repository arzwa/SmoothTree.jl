using Pkg; Pkg.activate(@__DIR__)
using SmoothTree, Test, NewickTree
using BenchmarkTools, Serialization, StatsBase, Distributions

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
    
    @testset "CCD benchmark" begin
        using SmoothTree: randtree
        @btime ccd = CCD(trees);
        # 33.679 ms (165153 allocations: 2.95 MiB)
        @btime tree = randtree(ccd);
        # 595.136 ns (20 allocations: 1.29 KiB)
        tree = randtree(ccd);
        @btime logpdf(ccd, tree);
        # 237.594 ns (1 allocation: 16 bytes)
    end
end
