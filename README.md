# SmoothTree

This library implements:

- Larget's **conditional clade distribution** (CCD) [@larget2013].
  This is a distribution over cladograms (labeled phylogenetic trees
  without meaningful branch lengths) derived from a collection of
  observed trees `X` under the assumption of conditional independence
  of disjoint subtrees. It is the maximum entropy distribution over
  tree topologies subject to the constraint of matching observed
  marginal split frequencies in `X` (see @szollosi2013).
- Aldous' **beta-splitting distribution** over cladograms
  [@aldous1996]. 
- Arbitrary **Markov branching models** (MBMs), of which the above are
  special cases. Crucially, the library implements a sparsely
  represented MBM that is obtained as the posterior of a
  beta-splitting (Dirichlet) prior distribution and an observed
  collection of splits (represented by a CCD). This can be seen as a
  smoothed CCD, which spans the whole tree space (the CCD does not
  generally cover the whole tree space)
- Efficient simulation routines for the **multi-species coalescent**
  (MSC) model.
- A likelihood-free **expectation propagation algorithm**
  [@barthelme2014] for approximate Bayesian inference of species trees
  from gene tree distributions.
  
## Examples

```julia
Coming soon.
```

## References

[@larget2013] Larget, Bret. "The estimation of tree posterior
probabilities using conditional clade probability distributions."
Systematic biology 62.4 (2013): 501-511.

[@aldous1996] Aldous, David. "Probability distributions on
cladograms." Random discrete structures. Springer, New York, NY, 1996.
1-18.

[@barthelme2014] Barthelmé, Simon, and Nicolas Chopin. "Expectation
propagation for likelihood-free inference." Journal of the American
Statistical Association 109.505 (2014): 315-333.
