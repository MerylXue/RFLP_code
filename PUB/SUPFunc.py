"""
Code for "Hao Shen, Mengying Xue, and Zuojun Max Shen (2023),
Data-driven Reliable Faciltiy Location Problem, Management Science, Forthcoming",

Last Revised on Jan 12, 2024
@author: Mengying Xue
"""

from Utility.Utils import DistSort
#
def OperCostCov(i, subset, num_cov, delta_marginal_prob_cov, delta_zeta_cond_i, lambda_i, num_J, d, wc_d):
    seq = DistSort(i, subset, d)
    seq.append(num_J)

    cost = d[i][seq[0]] * WC_Expected_Demand_cov([], num_J, num_cov, delta_marginal_prob_cov,
                                                                           delta_zeta_cond_i,
                                                                           wc_d)


    cost += sum((d[i][seq[r]] - d[i][seq[r - 1]]) * WC_Expected_Demand_cov([seq[l] for l in range(r)], num_J,
                                                                           num_cov, delta_marginal_prob_cov,
                                                                           delta_zeta_cond_i,
                                                                           wc_d) for r in range(1, len(seq)))

    cost += d[i][num_J] * sum([delta_marginal_prob_cov[k] * lambda_i[k] for k in range(num_cov)])
    return cost

def WC_Expected_Demand_cov(sigma, num_J, num_cov, delta_marginal_prob_cov, delta_zeta_cond_i, wc_d):

    key = [0 if i in sigma else 1 for i in range(num_J)]
    key = int("".join(map(str, key)), 2)

    # key = tuple(sorted(sigma))

    if key in wc_d.keys():
        value = wc_d[key]
    else:
        value = sum(delta_marginal_prob_cov[k] * delta_zeta_cond_i[k][a]
                                            for k in range(num_cov)
                                            for a in delta_zeta_cond_i[k].keys() if a & key == a)
        wc_d[key] = value

    return value



def TotalOperCostCov(subset, num_I, num_cov, delta_marginal_prob_cov, delta_zeta_cond, lambda_, num_J, d,  wc_d):
    total_oper_cost = sum(OperCostCov(i, subset, num_cov, delta_marginal_prob_cov, delta_zeta_cond[i] ,lambda_[i], num_J, d, wc_d[i])
                          for i in range(num_I))
    return total_oper_cost


def TotalFixedCost(subset, f):
    return sum(f[j] for j in subset)