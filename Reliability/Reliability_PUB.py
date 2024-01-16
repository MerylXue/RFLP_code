"""
Code for "Hao Shen, Mengying Xue, and Zuojun Max Shen (2023),
Data-driven Reliable Faciltiy Location Problem, Management Science, Forthcoming",

Last Revised on Jan 12, 2024
@author: Mengying Xue

"""


from Utility.Utils import BootstrapSample
from Utility.Constants import Right
from SampleAverage.SAAFunc import SAACost
from PUB.SUPDataDriven import SupLazy_AggragateCov

from DataRelated.DataProcess import SceProb
from copy import deepcopy
import math
from DataRelated.DataProcess import NumSamples
from Utility.Constants import converge_diff, Num_reliability

##Given the radius, return the reliability, out-of-sample cost, and the computational time
def BinarySearchEps_SupLazy(info, eps_group, rawdata, pre_data_process):
    t_search = 0
    data_driven_cost = []
    out_of_sample_cost = []
    Reliability = 0.0
    eps = eps_group[0]
    eps_samples = eps_group[1:]

    num_I = info['num_customer']
    f = info['fixed_cost']
    dist = info['dist']

    for k in range(Num_reliability):
        ## given the data, bootstrap the train data and set the remaining as the test data
        train_data, test_data = BootstrapSample(rawdata)
        #train data from bootstrap to optimize
        delta_marginal_prob_cov, delta_zeta_cond, lambda_ = SceProb(train_data, info['num_cov'], num_I, eps, eps_samples,
                                                           pre_data_process)

        loc_sup, obj_sup, t_sup, gap_sup = SupLazy_AggragateCov(info, delta_marginal_prob_cov, delta_zeta_cond, lambda_)
        t_search += t_sup

        saa_cost = SAACost(loc_sup, f, dist, num_I, info['max_demand'], test_data)
        data_driven_cost.append(obj_sup)
        out_of_sample_cost.append(saa_cost)

        ## if the out-of-sample cost is bounded by the data-driven cost
        if saa_cost <= obj_sup:
            Reliability += 1
    Reliability = Reliability/Num_reliability
    return Reliability, out_of_sample_cost, t_search


## generate theoretical radius (see Section 4.2)
def GenerateEpsCoeff(info, rawdata,  beta_):
    num_cov = info['num_cov']
    num_I = info['num_customer']
    num_J = info['num_facility']

    L = num_cov + 1
    num_data_samples = NumSamples(rawdata, num_cov)
    num_data = len(rawdata)
    eps_coeff = [0 for k in range(num_cov + 1)]
    eps_coeff[0] = math.sqrt(math.log((num_data+1)*L/beta_)/(2*num_data)) /num_cov
    for k in range(1, num_cov+1):
        if num_data_samples[k-1] > 0:
            eps_coeff[k] = math.sqrt(
                math.log((num_I + num_J) * (num_data_samples[k - 1] + 1) * (num_cov*L/(L-1)) / beta_) / (
                            2 * num_data_samples[k - 1]))
        else:
            eps_coeff[k] = 1
    return eps_coeff

def GenerateDirectionVector(num):
    vector = [(math.sin(math.pi/(2*num)*k), math.cos(math.pi/(2*num)*k)) for k in range(num+1)]
    return vector


def GenerateEpsByDirection(info, rawdata,  alpha1, alpha2):
    num_cov = info['num_cov']
    num_data_samples = NumSamples(rawdata, num_cov)
    num_data = len(rawdata)
    eps_coeff = [0 for k in range(num_cov + 1)]
    eps_coeff[0] = alpha1 * math.sqrt(1 / num_data)
    for k in range(1, num_cov+1):
        if num_data_samples[k-1] > 0:
            eps_coeff[k] = alpha2*math.sqrt(1/num_data_samples[k - 1])
        else:
            eps_coeff[k] = 1
    return eps_coeff

def RecoverEpsSamples(info, rawdata, beta_, eps):
    eps_coeff = GenerateEpsCoeff(info, rawdata, beta_)
    ratio = eps/eps_coeff[0]
    eps_samples = [ratio * eps_coeff[k] for k in range(1,len(eps_coeff))]
    return eps_samples


## find the radius by binary search in multi-covariates case
def GetOptEps_PUB_Ratio(info, beta_, rawdata, pre_data_process):
    left= 0
    eps_coeff = GenerateEpsCoeff(info, rawdata, beta_)
    right = Right/max(eps_coeff)
    opt_eps = -1
    # avg_out_of_sample_lst = []
    t_eps = 0
    count_eps_cov = 0
    Reliability = [0.0 for i in range(3)]
    out_of_sample_lst = [[] for i in range(3)]
    opt_reliability = 0
    left_samples = [min(1,left * eps_coeff[k]) for k in range(len(eps_coeff))]
    Reliability[0], tmp_list, t_search = BinarySearchEps_SupLazy(info, left_samples, rawdata, pre_data_process)
    out_of_sample_lst[0] = deepcopy(tmp_list)

    t_eps += t_search
    right_samples = [min(1,right * eps_coeff[k]) for k in range(len(eps_coeff))]
    Reliability[2], tmp_list, t_search = BinarySearchEps_SupLazy(info, right_samples, rawdata, pre_data_process)
    out_of_sample_lst[2] = deepcopy(tmp_list)
    t_eps += t_search

    while (right - left)*max(eps_coeff) >= converge_diff:
        count_eps_cov += 1
        mid = (right + left)/2
        mid_samples = [min(1,mid*eps_coeff[k]) for k in range(len(eps_coeff))]
        Reliability[1], tmp_list, t_search = BinarySearchEps_SupLazy(info, mid_samples, rawdata, pre_data_process)
        t_eps += t_search

        out_of_sample_lst[1] = deepcopy(tmp_list)
        if Reliability[1] >= 1 - beta_:
            opt_eps = mid
            avg_out_of_sample_lst = out_of_sample_lst[1]
            opt_reliability = Reliability[1]
            right = mid
            Reliability[2] = Reliability[1]
            Reliability[1] = 0.0
            out_of_sample_lst[2] = deepcopy(out_of_sample_lst[1])
            out_of_sample_lst[1] = []
        else:
            if Reliability[2] >= 1 - beta_:
                opt_eps = right
                avg_out_of_sample_lst = out_of_sample_lst[2]
                opt_reliability = Reliability[2]
                left = mid

                mid = right
                # ensure that the middle point is at the right
                Reliability[0] = Reliability[1]
                Reliability[1] = 0.0
                out_of_sample_lst[0] = deepcopy(out_of_sample_lst[1])
                out_of_sample_lst[1] = []
            else:
                break

    return opt_eps*eps_coeff[0], [opt_eps * eps_coeff[k] for k in range(1,len(eps_coeff))], opt_reliability, t_eps, count_eps_cov


# search for the radius in no-covaraite case
def GetOptEps_PUB_NoCov(info, beta_, rawdata, pre_data_process):
    left = 0
    right = Right
    opt_eps = -1
    opt_reliability = 0
    t_eps = 0
    count_eps_cov = 0
    Reliability = [0.0 for i in range(3)]
    out_of_sample_lst = [[] for i in range(3)]

    Reliability[0], tmp_list, t_search = BinarySearchEps_SupLazy(info, [0, left], rawdata, pre_data_process)
    out_of_sample_lst[0] = deepcopy(tmp_list)

    t_eps += t_search

    Reliability[2], tmp_list, t_search = BinarySearchEps_SupLazy(info, [0, right], rawdata, pre_data_process)
    out_of_sample_lst[2] = deepcopy(tmp_list)
    t_eps += t_search

    while (right - left) >= converge_diff:
        count_eps_cov += 1
        mid = (right + left)/2
        Reliability[1], tmp_list, t_search = BinarySearchEps_SupLazy(info, [0, mid], rawdata, pre_data_process)
        t_eps += t_search
        out_of_sample_lst[1] = deepcopy(tmp_list)
        if Reliability[0] > 1 - beta_:
            opt_eps = left
            # avg_out_of_sample_lst = out_of_sample_lst[0]
            break
        else:
            if Reliability[1] >= 1 - beta_:
                opt_eps = mid
                # avg_out_of_sample_lst = out_of_sample_lst[1]
                opt_reliability = Reliability[1]
                # right = mid
                right = mid
                Reliability[2] = Reliability[1]
                Reliability[1] = 0.0
                out_of_sample_lst[2] = deepcopy(out_of_sample_lst[1])
                out_of_sample_lst[1] = []
            else:
                if Reliability[2] >= 1 - beta_:
                    opt_eps = right
                    # avg_out_of_sample_lst = out_of_sample_lst[2]
                    left = mid
                    mid = right
                    opt_reliability = Reliability[2]
                    # ensure that the middle point is at the right
                    Reliability[0] = Reliability[1]
                    Reliability[1] = 0.0
                    out_of_sample_lst[0] = deepcopy(out_of_sample_lst[1])
                    out_of_sample_lst[1] = []
                else:
                    break

    return 0, [opt_eps], opt_reliability, t_eps, count_eps_cov


