from Utility.Utils import DistSortDisrupt

def OperCostData(i, subset, demand, xi, num_J, d):
    sigma = list(subset) + [num_J]
    loc_assign, distance = DistSortDisrupt(i,sigma, d, xi)

    # cost = (max_demand - demand) * distance
    cost = demand * distance
    return cost


## sample average cost
def SAACost(loc_chosen, f, d, num_I, max_demand, test_data):

    fixed_cost = sum([f[j] for j in loc_chosen])

    # # print(test_data)
    # cov_set = list(set(test_data['cov']))
    # max_demand = [[0 for k in range(len(cov_set))] for i in range(num_I)]
    # for k in range(len(cov_set)):
    #     data_k = test_data[test_data['cov'] == cov_set[k]]
    #     for i in range(num_I):
    #         # print(data_k.loc['d_%d'%i])
    #         max_demand[i][k] = data_k['d_%d'%i].max()


    oper_cost_total = 0
    for n in range(len(test_data)):
        demand = test_data.iloc[n,0:num_I].tolist()
        xi = test_data.iloc[n,num_I:num_I+num_I].tolist()
        oper_cost = sum([OperCostData(i, loc_chosen,  demand[i], xi, num_I, d) for i in range(num_I)])
        oper_cost_total += oper_cost

    return fixed_cost + oper_cost_total/len(test_data)