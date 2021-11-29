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
        #@btime ccd = CCD(trees);
        # 33.679 ms (165153 allocations: 2.95 MiB)
        #@btime splits = randsplits(ccd);
        # 595.136 ns (20 allocations: 1.29 KiB)
        #splits = randsplits(ccd);
        #@btime logpdf(ccd, splits);
        # 237.594 ns (1 allocation: 16 bytes)
    end

    @testset "MSC" begin
        using SmoothTree: randtree, isisomorphic
        S = nw"(((((A,B),C),(D,E)),(F,(G,H))),O);"
        SmoothTree.setdistance!(S, Inf)
        @test isisomorphic(randtree(MSC(S)), S)
        S = nw"((A,B),C);"
        SmoothTree.setdistance!(S, 0.)
        trees = proportionmap(randsplits(MSC(S), 1e5))
        @test all(values(trees) .- 1/3 .< 0.01)
    end

    @testset "Prior/regularization" begin
        using SmoothTree: initccd
        S = nw"(((A,B),C),D);"
        ccd = initccd(S, UInt8, 1.)  # an empty CCD
        @test ccd.cmap[maximum(keys(ccd.cmap))] == 0
        trees = proportionmap(randtree(ccd, 10000))
        # there should be 15 trees, three balanced ones and 12
        # unbalanced ones
        @test length(trees) == 15
        for (k,v) in trees
            nl(k) = length(getleaves(k))
            expected = nl(k[1]) == nl(k[2]) ? 0.15 : 0.05
            @test expected ≈ v atol=0.01
        end
    end

    @testset "logpdf" begin
        # we test whether the probability distribution sums to one
        using SmoothTree: initccd
        S = nw"(((A,B),C),D);"
        # prior distribution sums to one
        ccd = initccd(S, UInt8, 1.)  # an empty CCD
        trees = unique(randtree(ccd, 10000))
        @test length(trees) == 15
        p = mapreduce(tree->exp(logpdf(ccd, tree)), +, trees)
        @test p ≈ 1
        # known tree gives P = 1
        ccd = CCD(S, α=0.)
        @test logpdf(ccd, randtree(ccd)) ≈ 0.
        # posterior sums to one
        ccd = CCD(S, α=.5)
        trees = unique(randtree(ccd, 10000))
        p = mapreduce(tree->exp(logpdf(ccd, tree)), +, trees)
        @test p ≈ 1.
    end

    @testset "CCD from MSC sims" begin
        S = nw"((smo,(((gge,iov),(xtz,dzq)),sgt)),jvs);"
        θ = 1.
        SmoothTree.setdistance!(S, θ)
        model = MSC(S)
        data1 = CCD(model, randsplits(model, 10), α=1e-2)
        data2 = CCD(model, randsplits(model, 10), α=1e-2)
        SmoothTree.symmkldiv(data1, data2)
    end

end
