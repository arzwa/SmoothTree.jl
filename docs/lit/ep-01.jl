using Pkg; Pkg.activate(@__DIR__)
using SmoothTree, EPABC, NewickTree, LinearAlgebra, Distributions, StatsBase
using NewickTree: readnex

# model definition
struct MSCModel{M}
    model::M
end

function (model::MSCModel)(θ::Vector)
    SmoothTree.setdistance_internal_rooted!(model.model.tree, exp.(θ))
    return model
end

getmodel(model::MSCModel, ccd::CCD) = MSCModel(model.model(ccd))
Base.rand(model::MSCModel) = randsplits(model.model)

# input: true gene trees
S = readnex("docs/rev/sptree-01.nex")[1]
n = SmoothTree.n_internal(S) - 1
trees = readnex("docs/rev/genetrees-01.nex")[1:100]
data  = CCD.(trees)
last.(sort(collect(countmap(trees)), by=last))

μ = zeros(n)
Σ = diagm(fill(2., n)) 
M = 100000
accfun(ϵ=1.) = (x,y)->log(ϵ*rand()) < logpdf(x,y)
model = MSCModel(MSC(S))
alg = GaussianEPABC(data, model, accfun(), μ, Σ, M, α=0.1)

trace = ep!(alg, 1)

# plot
using Plots, StatsPlots, LaTeXStrings, Measures
mtrace = exp.(permutedims(mapreduce(x->x.μ, hcat, trace)))
#mtrace = mtrace[1:120,:]
p1 = plot(mtrace, xlabel=L"n", ylabel=L"\log\theta")
m = trace[end].μ
v = sqrt.(diag(trace[end].Σ))
p2 = plot(grid=false, legend=false, xlabel=L"\theta", ylabel="density")
for i=1:n
    plot!(Normal(m[i], v[i]))
end
plot(p1, p2, size=(500,200), layout=(1,2), bottom_margin=4mm,
     left_margin=4mm)


# input: inferred gene trees
using Serialization
S = readnex("docs/rev/sptree-01.nex")[1]
n = SmoothTree.n_internal(S) - 1

trees = map(readdir("docs/rev/ufboot-01/", join=true)) do fname
    trees = readnw.(readlines(fname))
    trees = getroot.(SmoothTree.rootall!(trees, "O"))
    proportionmap(trees)
end

serialize("docs/rev/ufboot-01.jls", trees)

data = map(CCD, trees)

μ = zeros(n)
Σ = diagm(fill(2., n)) 
M = 100000
accfun(ϵ=1.) = (x,y)->log(ϵ*rand()) < logpdf(x,y)
model = MSCModel(SmoothTree.MSC(S))
N = 500
alg = GaussianEPABC(data[1:N], model, accfun(0.1), μ, Σ, M, α=0.2)

trace = ep!(alg, 3)
trace = [trace; ep!(alg, 2)]

using Plots, StatsPlots, LaTeXStrings, Measures
Plots.default(legend=false, gridstyle=:dot, framestyle=:box)
mtrace = exp.(permutedims(mapreduce(x->x.μ, hcat, trace)))
#mtrace = mtrace[1:120,:]
p1 = plot(mtrace, xlabel=L"n", ylabel=L"\theta")
m = trace[end].μ
v = sqrt.(diag(trace[end].Σ))
p2 = plot(grid=false, legend=false, xlabel=L"\log\theta", ylabel="density")
for i=1:n
    plot!(Normal(m[i], v[i]))
end
timetree = readnex("docs/rev/sptree-01.nex")[1]
b = Dict(id(n)=>distance(n) for n in postwalk(model.model.tree))
xs = map(zip(postwalk(timetree), postwalk(coaltree))) do (node, c)
    if isleaf(node) || isroot(node) || isroot(parent(node))
        (NaN, NaN)
    else
        (distance(node), b[id(node)])
    end
end
# we expect a linear relationship between t and the coal branch
# lengths here (effective popsizes are all the same, only time
# differs).
p3 = scatter(filter(x->!isnan(x[1]), xs), size=(300,300),
             color=:lightgray, xlabel=L"t", ylabel=L"\theta") 
plot(p1, p2, p3, size=(700,200), layout=(1,3), bottom_margin=4mm,
     left_margin=4mm)

