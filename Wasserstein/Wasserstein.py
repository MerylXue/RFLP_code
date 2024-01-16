"""
Code for "Hao Shen, Mengying Xue, and Zuojun Max Shen (2023),
Data-driven Reliable Faciltiy Location Problem, Management Science, Forthcoming",

Last Revised on Jan 12, 2024
@author: Mengying Xue

"""

import numpy as np
from gurobipy import *
import time
from Utility.Utils import positive_num

## the optimization using wasserstein DRO
## The equivalent MILP formulation refers to Xie (2020)
## eps: the ambiguity size of Wassserstein ball
def WassersteinOpt(rawdata, info, eps):
    num_data = len(rawdata)
    num_I = info['num_customer']
    num_J = info['num_facility']
    f = info['fixed_cost']
    d = info['dist']
    max_demand = np.max(rawdata.to_numpy(), axis=0)[0:num_I]

    demand = [[ data[i] for i in range(num_I)] for data in rawdata.to_numpy()]
    disruption = [data[num_I:num_I+num_J] for data in rawdata.to_numpy()]

    I = range(num_I)
    J = range(num_J)

    model = Model('Wasserstein')
    model.Params.OutputFlag = False
    st = time.time()
    model.setParam(GRB.Param.TimeLimit, 900)
    x = model.addVars(num_J, vtype=GRB.BINARY)
    y = model.addVars(num_data, num_I, num_J + 1, lb=0)

    model.addConstrs((quicksum(y[l,i,j] for j in range(num_J+1)) == 1 for l in range(num_data) for i in I))
    model.addConstrs((y[l,i,j] <= positive_num(eps, 1) * disruption[l][j] * x[j] for l in range(num_data) for i in I for j in J))
    model.addConstrs((y[l,i,num_J]) <= 1 for l in range(num_data) for i in I)
    obj = quicksum(f[j] * x[j] for j in J)


    obj.add(quicksum((demand[l][i] + eps * max_demand[i])/ num_data * d[i][j] * y[l, i, j]for l in range(num_data)
                 for i in I for j in range(num_J + 1)))

    model.setObjective(obj, GRB.MINIMIZE)
    model.optimize()
    end = time.time()
    loc_chosen = [j for j in J if x[j].x > 0.5]
    obj_value = model.getObjective().getValue()
    gap = model.MIPGap
    sol_time = end-st
    return loc_chosen, obj_value,  sol_time, gap