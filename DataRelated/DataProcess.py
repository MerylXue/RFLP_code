"""
Code for "Hao Shen, Mengying Xue, and Zuojun Max Shen (2023),
Data-driven Reliable Faciltiy Location Problem, Management Science, Forthcoming",

Last Revised on Jan 12, 2024
@author: Mengying Xue
"""


import numpy as np
from Utility.Utils import diff_marginal_prob_cov


## generate marginal emp_prob_cov

##Data format
## rawdata: pandas, for each l: 0~I-1: demand for i, I~ I+J-1: state of location, I+J: covariate (discrete)

# Return the covariates in data set
def GroupDatabyCovariate(rawdata):
    data = rawdata.groupby('cov')
    # num_cov = len(set(rawdata['cov']))
    cov_set = list(set(rawdata['cov']))
    return data,  cov_set

# Return the data size
def NumSamples(rawdata, num_cov):
    data = rawdata.groupby('cov')

    cov_set = list(set(rawdata['cov']))
    num_data_sample = [0 for k in range(num_cov)]

    for k in range(num_cov):
        if k+1 in cov_set:
            item = data.get_group(k+1).to_numpy()
            num_data_sample[k] = len(item)
        else:
            num_data_sample[k] = 0

    return num_data_sample
#
# def MaxNumDisrupt(dictionary, eps):
#     temp = [len(x) for x in dictionary.keys() if dictionary[x] > eps]
#     if len(temp) == 0:
#         max_num = 0
#     else:
#         max_num = max(temp)
#     return max_num

## Reformulate the data in the format required by the optimization function
def PreProcess(rawdata, num_cov, num_I):
    I = range(num_I)
    # J = range(num_I)
    num_data = len(rawdata)

    data, cov_set = GroupDatabyCovariate(rawdata)

    num_data_sample = [0 for k in range(num_cov)]

    demand_val = [[[] for k in range(num_cov)] for i in I]
    emp_prob_demand = [[[] for k in range(num_cov)] for i in I]
    max_zeta = [[0 for k in range(num_cov)] for i in I]
    min_zeta = [[0 for k in range(num_cov)] for i in I]
    # calculate the marginal probabilities conditional on the covariate
    marginal_prob_cov = np.zeros(num_cov)

    for k in range(num_cov):
        if k + 1 in cov_set:
            item = data.get_group(k + 1).to_numpy()

            num_data_sample[k] = len(item)
            marginal_prob_cov[k] = len(item) / num_data
            # summarize possible demand values (increasing order) for each customer
            demand_val_tmp = [list(np.unique([-each[i] for each in item])) for i in I]

            for i in I:
                max_zeta[i][k] = max(demand_val_tmp[i])
                min_zeta[i][k] = min(demand_val_tmp[i])
                temp = [(n, demand_val_tmp[i][n]) for n in range(len(demand_val_tmp[i]))]
                temp_sort = sorted(temp, key=lambda a: a[1])
                demand_val[i][k] = [a[1] for a in temp_sort]


            for i in I:
                emp_prob_demand[i][k] = [0 for m in range(len(demand_val[i][k]))]
            for each in item:
                for i in I:
                    for m in range(len( demand_val[i][k])):
                        if demand_val[i][k][m] >= -each[i]:
                            emp_prob_demand[i][k][m] += 1.0 / num_data_sample[k]



    cdf_prob_cov = [sum(marginal_prob_cov[:k + 1]) for k in range(num_cov - 1)] + [sum(marginal_prob_cov)]
    pre_data_process = {}
    pre_data_process['cdf_prob_cov'] = cdf_prob_cov
    pre_data_process['emp_prob_demand'] = emp_prob_demand
    pre_data_process['demand_val'] = demand_val
    pre_data_process['num_data_sample'] = num_data_sample
    pre_data_process['max_zeta'] = max_zeta
    pre_data_process['min_zeta'] = min_zeta
    return pre_data_process



# return the covariate marginal probabilities vector
def DeltaMarginalProbCov(cdf_prob_cov, eps):
    delta_marginal_prob_cov = [min(1, cdf_prob_cov[0] + eps)] + [diff_marginal_prob_cov(cdf_prob_cov, k, eps) for k in
                                                                 range(1, len(cdf_prob_cov))]
    return delta_marginal_prob_cov

def ZetaThreshold(emp_prob_demand_i_k, demand_val_i_k, eps_k, max_zeta_i_k, min_zeta_i_k):
    for m in range(len(demand_val_i_k)):
        if emp_prob_demand_i_k[m] > 1-eps_k:
            return demand_val_i_k[m], max_zeta_i_k -eps_k * min_zeta_i_k - demand_val_i_k[m]*(1-eps_k)
    return max_zeta_i_k, (max_zeta_i_k - min_zeta_i_k) * eps_k

def ConditionalZeta(data_k, i, num_I, zeta_i_k,  num_samples_k):
    sorted_item = sorted(data_k, key=lambda a: sum(a[num_I:num_I + num_I]))
    J=range(num_I)
    delta_zeta_cond_i_k = {}
    for each in sorted_item:
        delta_demand = max(zeta_i_k + each[i], 0)
        temp = int("".join(map(str, [int(each[num_I + j]) for j in J])), 2)
        if temp in delta_zeta_cond_i_k.keys():
            delta_zeta_cond_i_k[temp] += delta_demand /num_samples_k
        else:
            delta_zeta_cond_i_k[temp] = delta_demand / num_samples_k

    return delta_zeta_cond_i_k

## generate the scenario based data
def SceProb(rawdata, num_cov, num_I, eps, eps_samples, pre_data_process):
    cdf_prob_cov = pre_data_process['cdf_prob_cov']
    emp_prob_demand = pre_data_process['emp_prob_demand']
    demand_val = pre_data_process['demand_val']
    num_data_sample = pre_data_process['num_data_sample']
    max_zeta = pre_data_process['max_zeta']
    min_zeta = pre_data_process['min_zeta']
    delta_marginal_prob_cov = DeltaMarginalProbCov(cdf_prob_cov, eps)
    delta_zeta_cond = [[{} for k in range(num_cov)] for i in range(num_I)]
    lambda_ = [[0 for k in range(num_cov)] for i in range(num_I)]
    for k in range(num_cov):
        data_k = rawdata[rawdata['cov'] == k+1].to_numpy()
        for i in range(num_I):
            zeta_threshold, lambda_tmp = ZetaThreshold(emp_prob_demand[i][k], demand_val[i][k], eps_samples[k],
                                                       max_zeta[i][k], min_zeta[i][k])

            lambda_[i][k] = lambda_tmp
            delta_zeta_cond[i][k] = ConditionalZeta(data_k, i, num_I, zeta_threshold, num_data_sample[k])


    return delta_marginal_prob_cov, delta_zeta_cond, lambda_
