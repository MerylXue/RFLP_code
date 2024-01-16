from Wasserstein.Wasserstein import WassersteinOpt
from copy import deepcopy
from Utility.Constants import Right
import math
from DataRelated.DataProcess import NumSamples
from Utility.Utils import BootstrapSample
from SampleAverage.SAAFunc import SAACost
from Utility.Constants import converge_diff, Num_reliability, search_step
import time
def BinarySearchEps_Wasserstein(info, eps, rawdata):
    t_search = 0
    data_driven_cost = []
    out_of_sample_cost = []
    Reliability = 0.0

    num_I = info['num_customer']
    f = info['fixed_cost']
    dist = info['dist']

    for k in range(Num_reliability):
        train_data, test_data = BootstrapSample(rawdata)
        loc_sup, obj_sup, t_sup, gap_sup = WassersteinOpt(train_data, info, eps)
        t_search += t_sup
        saa_cost = SAACost(loc_sup, f, dist, num_I, info['max_demand'], test_data)

        data_driven_cost.append(obj_sup)
        out_of_sample_cost.append(saa_cost)

        if saa_cost <= obj_sup:
            Reliability += 1
    Reliability = Reliability/Num_reliability
    # print(sum(data_driven_cost)/Num_reliability, sum(out_of_sample_cost)/Num_reliability)
    return Reliability, out_of_sample_cost, t_search


def GetOptEpsWasser(info, beta_, rawdata):
    left = 0.0
    num_I = info['num_customer']
    right = Right
    opt_eps = -1
    avg_out_of_sample_lst = []
    t_eps = 0
    count_eps = 0
    Reliability = [0.0 for i in range(3)]
    out_of_sample_lst = [[] for i in range(3)]


    Reliability[0], tmp_list, t_search = BinarySearchEps_Wasserstein(info, left, rawdata)
    out_of_sample_lst[0] = deepcopy(tmp_list)
    # print("initial left---------------")
    # print(Reliability[0], left)

    t_eps += t_search
    Reliability[2], tmp_list, t_search = BinarySearchEps_Wasserstein(info, right, rawdata)
    out_of_sample_lst[2] = deepcopy(tmp_list)
    # print("initial right-------------------")
    # print(Reliability[2], right)
    t_eps += t_search
    ## set the time limit for searching optimal eps as 750
    while (right - left) >= converge_diff and t_eps < 750:
        mid = (right + left) / 2
        Reliability[1], tmp_list, t_search = BinarySearchEps_Wasserstein(info, mid, rawdata)
        out_of_sample_lst[1] = deepcopy(tmp_list)
        # print("mid--------------------------")
        # print(Reliability[1], mid)
        t_eps += t_search
        count_eps += 1

        if Reliability[1] >= 1 - beta_:
            # print("mid-----------------------")
            opt_eps = mid
            avg_out_of_sample_lst = out_of_sample_lst[1]
            right = mid
            Reliability[2] = Reliability[1]
            Reliability[1] = 0.0
            out_of_sample_lst[2] = deepcopy(out_of_sample_lst[1])
            out_of_sample_lst[1] = []

        else:
            if Reliability[2] >= 1 - beta_:
                # print("right-------------------")
                opt_eps = right
                avg_out_of_sample_lst = out_of_sample_lst[2]
                left = mid
                # mid = right
                Reliability[0] = Reliability[1]
                Reliability[1] = 0.0
                out_of_sample_lst[0] = deepcopy(out_of_sample_lst[1])
                out_of_sample_lst[1] = []
            else:
                print("right reliability %f, convergence %f"%(Reliability[2], right - left))
                opt_eps = right
                avg_out_of_sample_lst = out_of_sample_lst[2]
                break
    print("Find the eps for Wasserstein-------------")
    # print(Reliability)
    print(opt_eps)
    # print(out_of_sample_lst)
    return opt_eps, avg_out_of_sample_lst, t_eps, count_eps
