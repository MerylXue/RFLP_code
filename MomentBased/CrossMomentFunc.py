"""
Code for "Hao Shen, Mengying Xue, and Zuojun Max Shen (2023),
Data-driven Reliable Faciltiy Location Problem, Management Science, Forthcoming",

Last Revised on Jan 12, 2024
@author: Mengying Xue

"""
from Utility.Utils import DistSort

## in cross moment method and data with no covariates, given the first and second order moment info
## calculate the second-stage cost for serving customer i
def OperCostCrossMomentNoCov(i, subset, MarginalProb, SecondMomentProb, IndexPair, mu_i, d):

    num_I = num_J = len(d)
    sigma = subset + [num_J]
    seq = DistSort(i,sigma, d)

    cost = d[i][seq[0]] * mu_i

    cost += sum([(d[i][seq[r + 1]] - d[i][seq[r]]) \
                * mu_i * WCQ_Prob(seq[:r + 1], MarginalProb, SecondMomentProb, IndexPair)
                for r in range(len(seq) - 1) ])

    return cost

## in cross moment method and data with covariates, given the first and second order moment info
## calculate the second-stage cost for serving customer i
def OperCostCrossMomentCov(i, subset, num_cov,  CovProb, MarginalProb, SecondMomentProb, IndexPair, mu_i, d):

    num_I = num_J = len(d)
    sigma = subset + [num_J]
    seq = DistSort(i,sigma, d)

    cost = d[i][seq[0]] * sum([mu_i[k] * CovProb[k] for k in range(num_cov)])

    cost += sum([(d[i][seq[r + 1]] - d[i][seq[r]]) \
                * mu_i[k] * CovProb[k] * WCQ_Prob(seq[:r + 1], MarginalProb[k], SecondMomentProb[k], IndexPair[k])
                for r in range(len(seq) - 1) for k in range(num_cov) if len(MarginalProb[k]) > 0])


    return cost

## in cross moment method, calculate the worst-case probability
def WCQ_Prob(sigma, MarginalProb, SecondMomentProb, IndexPair):
    min_prob = min([MarginalProb[loc] for loc in sigma])

    for idx_pair in range(len(IndexPair)):
        pair_loc = IndexPair[idx_pair]
        if pair_loc[0] in sigma and pair_loc[1] in sigma:
            min_prob = min(min_prob, SecondMomentProb[IndexPair.index(pair_loc)])

    return min_prob


