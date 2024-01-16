import math
from DataRelated.DataProcess import GroupDatabyCovariate
import numpy as np
from itertools import combinations



def GenerateMomentDataCov(info, rawdata, num_cov, num_I, num_J):
    # max_demand = info['max_demand']
    data, cov_set = GroupDatabyCovariate(rawdata)
    # k * i
    mean_demand_cond  =  [[] for k in range(num_cov)]
    num_data = len(rawdata)
    marginal_prob_cov = np.zeros(num_cov)
    num_data_sample = [0 for k in range(num_cov)]
    marginal_prob_disrupt = [[] for k in range(num_cov)]
    SecondMomentProb = [[] for k in range(num_cov)]
    IndexPair = [[] for k in range(num_cov)]
    # idx_cov = 0
    # for k in cov_set:
    # print("Marginal probability in covariate")
    for k in range(num_cov):
        if k + 1 in cov_set:
            item = data.get_group(k + 1).to_numpy()
            num_data_sample[k] = len(item)
            marginal_prob_cov[k] = len(item) / num_data
            # mean_demand_cond[k] = [np.average([max_demand[i]-each[i] for each in item])  for i in range(num_I)]
            mean_demand_cond[k] = [np.average([each[i] for each in item])  for i in range(num_I)]

            marginal_prob_disrupt[k] = [np.average([1-each[num_I+j] for each in item]) for j in range(num_J)]
            # print(marginal_prob_disrupt[k])


            SecondMomentProb[k], IndexPair[k] = DisruptMoment(num_I,num_J, num_data_sample[k], item[:,0:num_I+num_J])
        else:
            num_data_sample[k] = 0
            marginal_prob_cov[k] = 0
            mean_demand_cond[k] = [0 for i in range(num_I)]
            marginal_prob_disrupt[k] = [0 for j in range(num_J)]

        # idx_cov += 1
    return np.array(mean_demand_cond), marginal_prob_cov, marginal_prob_disrupt, SecondMomentProb, IndexPair


def GenerateMomentDataNoCov(info, rawdata, num_I, num_J):
    # data, num_cov, cov_set = GroupDatabyCovariate(rawdata)
    # k * i
    # max_demand = info['max_demand']
    num_data = len(rawdata)
    rawdata = rawdata.to_numpy()
    # mean_demand_cond = [np.average([max_demand[i]-each[i] for each in rawdata])  for i in range(num_I)]
    mean_demand_cond = [np.average([each[i] for each in rawdata])  for i in range(num_I)]

    marginal_prob_disrupt = [np.average([1-each[num_I+j] for each in rawdata]) for j in range(num_J)]
    SecondMomentProb, IndexPair= DisruptMoment(num_I,num_J, num_data, rawdata[:,0:num_I+num_J])

    return np.array(mean_demand_cond), marginal_prob_disrupt, SecondMomentProb, IndexPair


def DisruptMoment(num_I, num_J, num_data, rawdata):


    SecondMomentProb = []
    IndexPair = []
    DisruptPair = {}
    for l in range(num_data):
        disrupt_loc = []
        for j in range(num_I, num_I+num_J):
            if rawdata[l][j] == 0:
                disrupt_loc.append(j-num_I)
        if len(disrupt_loc) > 1:
            for item in list(combinations(disrupt_loc, 2)):
                if item in DisruptPair.keys():
                    val = DisruptPair[item]
                    val += 1
                    DisruptPair.update({item: val})
                else:
                    DisruptPair.update({item: 1})

    for key, val in DisruptPair.items():

        prob = val/num_data
        k0 = min(key[0], key[1])
        k1 = max(key[0], key[1])

        IndexPair.append([k0, k1])
        SecondMomentProb.append(prob)

    return SecondMomentProb, IndexPair