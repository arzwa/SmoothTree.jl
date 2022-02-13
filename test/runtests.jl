using Pkg; Pkg.activate(@__DIR__)
using SmoothTree, Test, NewickTree
using StatsBase, Distributions

@testset "SmoothTree tests" begin
    treesfile = joinpath(@__DIR__, "OG0006030.trees")
    #treesfile = joinpath(@__DIR__, "test/OG0006030.trees")
    trees = readnw.(readlines(treesfile))
    trees = SmoothTree.rootall!(trees)
    tmap  = clademap(trees)
    ccd   = CCD(trees, tmap)

    @testset "Beta-splitting SparseSplits" begin
        γ = UInt8(15)
        n = cladesize(γ)
        d = Dict{UInt8,Int}()
        # β = -3/2 -> PDA model (uniform onlabeled topologies)
        beta = BetaSplitTree(-1.5, n)
        b = SmoothTree.SparseSplits(γ, d, beta, 1.)
        m = SmoothTree.nat2mom(b)
        ps = m.η0 .* m.k
        @test sum(ps) == 1
        @test ps[1] / 12 ≈ ps[2] / 3 ≈ 1/15 
        # there are 15 topologies on 4 leaves, three of which are
        # balanced. with β=-1.5 we should get all topologies equally
        # likely. Each balanced split uniquely determines a topology,
        # so we should get p(balanced split) = 1/15
        # conditional on the first split being balanced, we have three
        # possible topologies, whereas conditional on the first split
        # being unbalanced we have 12. 
        # β = 0. -> Yule model (coalescent, random joins)
        b = SmoothTree.SparseSplits(γ, d, beta(0.), 1.)
        m = SmoothTree.nat2mom(b)
        ps = m.η0 .* m.k
        @test sum(ps) ≈ 1
        @test ps[1] ≈ 2/3 && ps[2] ≈ 1/3
        # under the coalescent, the first coalescence leads to a state
        # with one cherry and two leaves, the next coalescence is in
        # 2/3 cases joining the cherry with one of the leaves leading
        # to an unbalanced topology, in 1/3 cases joins the two leaves
        # to a second cherry, giving a balanced topology.
        d = Dict{UInt8,Int}(5=>2, 1=>1, 7=>4)
        b = SmoothTree.SparseSplits(γ, d, beta(0.), 1.)
        m = SmoothTree.nat2mom(b)
        p = sum(m.η0 .* m.k) + sum(values(m.splits))
        pr = sum(values(m.splits))
        @test p ≈ 1.
        b = SmoothTree.SparseSplits(γ, d, beta(0.), 10.)
        m = SmoothTree.nat2mom(b)
        @test sum(values(m.splits)) < pr
        b = SmoothTree.SparseSplits(γ, d, beta(-1.5), 10e8)
        m = SmoothTree.nat2mom(b)
        @test sum(values(m.splits)) * 15 ≈ 7.
        # with α very large we should be indistinguishable from the prior
        # which is uniform on topologies for β=-1.5, the three
        # represented root splits should represent 7/15 topologies
    end

    @testset "Beta split MBM, PDA test" begin
        for n = 4:10
            root = 2^n-1
            bs = BetaSplitTree(-1.5, cladesize(root))
            M = MomMBM(root, bs)
            sims = randsplits(M, 10000) |> unique 
            ls = logpdf.(Ref(M), sims)
            @test all(ls .≈ log(1/SmoothTree.ntrees(n)))
            # under β=-1.5, all trees are equally likely
        end
    end

    @testset "Posterior MBM" begin        
        bs = BetaSplitTree(-1., cladesize(ccd.root))
        M1 = MomMBM(ccd, bs, 1e-6)
        M2 = MomMBM(ccd, bs, 1.)
        for i=1:100
            x = randsplits(ccd)
            l1 = logpdf(M1, x)
            l2 = logpdf(M2, x)
            l3 = logpdf(ccd, x)
            @test l1 ≈ l3
            #@test l2 < l3  # why is this not always true?
        end
    end

    @testset "BranchModel algebra" begin
        q1 = BranchModel(UInt16, [1., -1.])
        q2 = BranchModel(UInt16, [2., -4.])
        q3 = 0.1q1 + 2.1q2
    end

    @testset "NatBMP algebra" begin 
        using SmoothTree: nat2mom, mom2nat
        S = nw"((B:Inf,C:Inf):0.5,A:Inf);"
        m = clademap(S)
        M = MSC(S, m)
        Y = randtree(M, m, 100)
        bsd = BetaSplitTree(-1., length(m))
        X = NatMBM(CCD(Y, m), bsd, 1.)
        # sparsesplits algebra
        y = X.smap[0x0007]
        z = mom2nat(nat2mom(0.2y + 0.8y))
        @test z.η0 ≈ y.η0
        @test all(values(z.splits) .≈ values(y.splits))
        X = NatMBM(CCD(Y[1:2], m), bsd, 1.)
        γ = 0x0007
        y = X.smap[γ]
        z = SmoothTree.SparseSplits(γ, Dict{UInt16,Int}(), bsd, 1.)
        @test (y + z).η0 ≈ y.η0
        @test (y - y).η0 ≈ z.η0
    end

    @testset "MSCModel" begin
        S = nw"((B:Inf,C:Inf):0.5,A:Inf);"
        m = clademap(S, UInt16)
        M = MSC(S, m)
        Y = randtree(M, m, 100)
        b = BetaSplitTree(-1., length(m))
        X = NatMBM(CCD(Y, m), b, 1.)
        q = BranchModel(UInt16(sum(keys(m))), [1., -1.])
        M1 = MSCModel(X, q)
        M2 = M1 + M1*0.3
    end

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
        m = clademap(S)
        init = SmoothTree.default_init(S, m)
        SmoothTree.setdistance!(S, Inf)
        @test isisomorphic(randtree(MSC(S, init), m), S, m)
        S = nw"((A,B),C);"
        m = clademap(S)
        init = SmoothTree.default_init(S, m)
        SmoothTree.setdistance!(S, 0.)
        trees = proportionmap(randsplits(MSC(S, init), 1e5))
        @test all(values(trees) .- 1/3 .< 0.01)
    end

    @testset "logpdf" begin
        # we test whether the probability distribution sums to one
        using SmoothTree: initccd
        S = nw"(((A,B),C),D);"
        m = clademap(S)
        # prior distribution sums to one
        bsd = BetaSplitTree(-1.5, length(m))
        mbm = MomMBM(sum(keys(m)), bsd)
        trees = unique(randsplits(mbm, 10000))
        @test length(trees) == 15
        p = mapreduce(tree->exp(logpdf(mbm, tree)), +, trees)
        @test p ≈ 1
        # known tree gives P = 1
        ccd = CCD(S, m)
        @test logpdf(ccd, randsplits(ccd)) ≈ 0.
        # posterior sums to one
        mbm = MomMBM(ccd, bsd, .5)
        trees = unique(randsplits(mbm, 10000))
        p = mapreduce(tree->exp(logpdf(mbm, tree)), +, trees)
        @test p ≈ 1.
    end

    @testset "Verify sampler with logpdf/logpdf with sampler" begin
        for α=[0.1, 1., 10.], β=[-1.99, -1.5, -1., 0., 1., Inf]
            α=0.1; β=-1.
            n = 100
            # we get some 'observed data' from MSC simulations
            S = nw"(((A,B),C),D);"
            m = clademap(S)
            bsd = BetaSplitTree(β, length(m))
            SmoothTree.setdistance!(S, 5.)
            model = MSC(S, SmoothTree.default_init(S, m))
            data = CCD(randtree(model, m, 1000), m)
            # we construct a BMP tree prior
            treeprior = MomMBM(CCD(randtree(data, m, n), m), bsd, α)
            # estimate the likelihood of the 15 trees using simulation
            trees = ranking(randsplits(treeprior, 100000))
            # compute the likelihood of the 15 trees algorithmically
            ls = exp.(logpdf.(Ref(treeprior), first.(trees)))
            ps = last.(trees)
            # compare
            #for i=1:15; @printf "%s %.4f %.4f\n" trees[i][1] trees[i][2] ls[i]; end
            @test all(isapprox(ls, ps, rtol=0.1))
        end
    end

    @testset "Tree isomorphism" begin
        t1 = nw"(((((gge,iov),(xtz,dzq)),sgt),smo),jvs);"
        t2 = nw"((smo,(((gge,iov),(xtz,dzq)),sgt)),jvs);"
        m = clademap(t1)
        @test SmoothTree.isisomorphic(t1, t2, m)
    end

end
