using Pkg; Pkg.activate(@__DIR__)
using SmoothTree, Test, NewickTree
using StatsBase, Distributions

@testset "SmoothTree tests" begin
    treesfile = joinpath(@__DIR__, "test/OG0006030.trees")
    #treesfile = joinpath(@__DIR__, "OG0006030.trees")
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
        using SmoothTree: randtree, isisomorphic, MSC
        S = nw"(((((A,B),C),(D,E)),(F,(G,H))),O);"
        m = taxonmap(S)
        init = SmoothTree.default_init(S, m)
        SmoothTree.setdistance!(S, Inf)
        @test isisomorphic(randtree(MSC(S, init), m), S, m)
        S = nw"((A,B),C);"
        m = taxonmap(S)
        init = SmoothTree.default_init(S, m)
        SmoothTree.setdistance!(S, 0.)
        trees = proportionmap(randsplits(MSC(S, init), 1e5))
        @test all(values(trees) .- 1/3 .< 0.01)
    end

    @testset "Prior/regularization" begin
        ccd = CCD(["A","B","C","D"], α=1.)
        @test ccd.cmap[maximum(keys(ccd.cmap))] == 0
        trees = proportionmap(randtree(ccd, 10000))
        # there should be 15 trees, three balanced ones and 12
        # unbalanced ones
        @test length(trees) == 15
        for (k,v) in trees
            nl(k) = length(getleaves(k))
            expected = nl(k[1]) == nl(k[2]) ? 3/21 : 1/21
            @test expected ≈ v atol=0.01
        end
    end

    @testset "logpdf" begin
        # we test whether the probability distribution sums to one
        using SmoothTree: initccd
        S = nw"(((A,B),C),D);"
        # prior distribution sums to one
        ccd = CCD(["A","B","C","D"], α=1.)  # an empty CCD
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

    @testset "Verify sampler with logpdf/logpdf with sampler" begin
        for α=[0.1, 0.5, 1., 2.]
            n = 1000
            # we get some 'observed data' from MSC simulations
            S = nw"(((A,B),C),D);"
            m = taxonmap(S)
            SmoothTree.setdistance!(S, 5.)
            model = SmoothTree.MSC(S, SmoothTree.default_init(S, m))
            #data = CCD(model, SmoothTree.randsplits(model, 1000), α=0.001) 
            data = SmoothTree.randtree(model, m, 1000)
            ccd = CCD(data, lmap=m, α=0.001) 
            # we construct a BMP tree prior
            treeprior = CCD(randtree(ccd, n), α=α*n)
            # estimate the likelihood of the 15 trees using simulation
            trees = SmoothTree.ranking(randtree(treeprior, 100000))
            # compute the likelihood of the 15 trees algorithmically
            ls = exp.(logpdf.(Ref(treeprior), first.(trees)))
            ps = last.(trees)
            # compare
            #for i=1:15; @printf "%s %.4f %.4f\n" trees[i][1] trees[i][2] ls[i]; end
            @test all(isapprox(ls, ps, rtol=0.2))
        end
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

    @testset "Tree isomorphism" begin
        t1 = nw"(((((gge,iov),(xtz,dzq)),sgt),smo),jvs);"
        t2 = nw"((smo,(((gge,iov),(xtz,dzq)),sgt)),jvs);"
        m = taxonmap(t1)
        @test SmoothTree.isisomorphic(t1, t2, m)
    end

    @testset "marginal clade size" begin
        using SmoothTree: cladesize, getcladesbits
        S = nw"((smo,(((gge,iov),(xtz,dzq)),sgt)),jvs);"
        # prior distribution sums to one
        ccd = SmoothTree.initccd(S, UInt8, 10.)  # an empty CCD
        trees = randtree(ccd, 100000)
        empirical = [cladesize.(getcladesbits(t)) for t in trees]
        map(1:7) do k
            p = length(filter(x->k ∈ x, empirical)) / length(empirical)
            @test SmoothTree._pcladesize(7,k) ≈ p rtol=0.1
        end
    end

end
