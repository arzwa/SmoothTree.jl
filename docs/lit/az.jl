# simulation experiment script for anomalyzone
using Pkg; Pkg.activate(@__DIR__)
using SmoothTree, NewickTree, StatsBase, Printf, Serialization, DataFrames, CSV

outdir = mkpath("docs/data/anomalyzone-out")

ntaxa = 5
config = (ntaxa=ntaxa, 
          θ=1/(ntaxa-1),
          N=100, 
          nparticle=50000,
          miness=5.,
          target=500.,
          β=-1.5,  
          μ=0.5,
          V=2.,
          npass=5,
          nrun=2,
          α=1e-3,
          λ=0.1,
         )

write(joinpath(outdir, "config.txt"), string(config))

# simulate a pectinate (β=-2) species tree
T = UInt16
o = rootclade(config.ntaxa, T) 
S = randtree(NatMBM(o, BetaSplitTree(-2., config.ntaxa))) 
l = SmoothTree.n_internal(S)
#S = nw"(E,(D,(C,(B,A))));"
SmoothTree.setdistance!(S, 1/(ntaxa-1))
m = clademap(S, T)

# simulate gene trees
M = SmoothTree.MSC(S, m)
N = config.N 
G = randtree(M, m, N)

# write the trees
function writetreed(pth, trees::Vector)
    open(pth, "w") do io
        for (t, v) in trees
            write(io, @sprintf("%6.4f %s\n", v, nwstr(t)))
        end
    end
end

writetreed(joinpath(outdir, "genetrees.txt"), ranking(G))

# data set 
#a = 1/2^(config.ntaxa)
bsd = BetaSplitTree(-1.5, ntaxa)
#data = Locus.(G, Ref(m), a, -1.5)
data = Locus.(G, Ref(m))

# priors
tips = collect(keys(m))
Sprior = NatMBM(o, bsd)
θprior = BranchModel(o, gaussian_mom2nat([log(config.μ), config.V]), inftips=tips)

# model/algorithm
model = MSCModel(Sprior, θprior)

results = map(1:config.nrun) do i
    alg = EPABCIS(data, model, config.nparticle, target=config.target,
                  miness=config.miness, λ=config.λ, α=config.α)
    trace = ep!(alg, config.npass)
    # check MAP tree
    smple = ranking(relabel.(randtree(alg.model.S, 10000), Ref(m)))
    truS  = SmoothTree.topologize(S)
    mapS  = smple[1][1]
    writetreed(joinpath(outdir, "posterior_S.txt"), smple)
    @info "True vs. MAP" truS mapS
    (S=truS, Smap=mapS, pmap=smple[1][2], Z=last(trace)[2])
end

# save the trace and posterior approximation
#xs = traceback(first.(trace))
#Zs = getindex.(trace, 2)
#serialize(joinpath(outdir, "trace.jls"), (trace=xs, Z=Zs))
#serialize(joinpath(outdir, "posterior.jls"), alg.model)

CSV.write(joinpath(outdir, "result.csv"), DataFrame(results))

