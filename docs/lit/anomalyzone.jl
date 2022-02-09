# simulation experiment script for anomalyzone
using Pkg; Pkg.activate(@__DIR__)
using SmoothTree, NewickTree, StatsBase, Printf, Serialization

outdir = mkpath("docs/data/anomalyzone-out")

ntaxa = 5
config = (ntaxa=ntaxa, 
          θ=1/(ntaxa-1),
          N=100, 
          h=1e3, 
          α=1e-2,
          λ=0.05,
          ν=0.05,
          β=-1.5,  
          target=100, 
          minacc=10,
          batch=100,
          μ=0.5,
          V=2.,
          npass1=3,
          npass2=2
         )

write(joinpath(outdir, "config.txt"), string(config))

# simulate a pectinate (β=-2) species tree
T = UInt16
o = rootclade(config.ntaxa, T) 
S = randtree(NatMBM(o, BetaSplitTree(-2., config.ntaxa))) 
l = SmoothTree.n_internal(S)
SmoothTree.setdistance!(S, 1/(ntaxa-1))
m = taxonmap(S, T)

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
a = 1/2^(config.ntaxa-1)
bsd = BetaSplitTree(-1.5, ntaxa)
data = CCD.(G, Ref(m))
data = MomMBM.(data, Ref(bsd), a)

# priors
#Sprior = NatMBM(CCD(unique(G), m), bsd, 100.)
Sprior = NatMBM(o, bsd)
θprior = BranchModel(o, gaussian_mom2nat([log(config.μ), config.V]))

# model/algorithm
model = MSCModel(Sprior, θprior, m)
alg = EPABC(data, model, prunetol=0., h=config.h, λ=config.λ, ν=config.ν, α=config.α,
            target=config.target, minacc=config.minacc, batch=config.batch)

# inference
trace = ep!(alg, config.npass1)
SmoothTree.tuneoff!(alg)
trace = [trace; ep!(alg, config.npass2)]

# check MAP tree
smple = randtree(alg.model.S, 10000)
smple = map(x->relabel(x, m), smple)
smple = ranking(smple)
truS  = SmoothTree.topologize(S)
mapS  = smple[1][1]
writetreed(joinpath(outdir, "posterior_S.txt"), smple)
@info "True vs. MAP" truS mapS

# save the trace and posterior approximation
X1, X2 = traceback(first.(trace))
serialize(joinpath(outdir, "trace.jls"), (S=X1, q=X2, Z=last.(trace)))
serialize(joinpath(outdir, "posterior.jls"), (model=alg.model, tmap=m))

