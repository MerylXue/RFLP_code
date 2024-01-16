"""
Code for "Hao Shen, Mengying Xue, and Zuojun Max Shen (2023),
Data-driven Reliable Faciltiy Location Problem, Management Science, Forthcoming",

Last Revised on Jan 12, 2024
@author: Mengying Xue

"""
import time
from gurobipy import *
from SampleAverage.SAAFunc import OperCostData

def SAA1(rawdata, info):
    num_I = info['num_customer']
    num_J = info['num_facility']
    I = range(num_I)
    J = range(num_J)
    d = info['dist']
    num_data = len(rawdata)
    rawdata = rawdata.to_numpy()
    f = info['fixed_cost']

    model = Model('SUP')
    model.modelSense = GRB.MINIMIZE

    tstart = time.time()
    x = model.addVars(num_J, vtype=GRB.BINARY, obj=f[:-1])
    gamma = model.addVars(num_I, obj=[1 for i in I])
    ini_sol = [j for j in J]

    model.addConstrs(
        (gamma[i] >= sum([OperCostData(i, ini_sol, rawdata[k][i], rawdata[k][num_I:num_I+num_J], num_J, d)
                              for k in range(num_data)])/num_data
         for i in I )
    )
    def mycallback(model, where):
        if where == GRB.Callback.MIPSOL:
            relg = model.cbGetSolution(model._varg)
            relx = model.cbGetSolution(model._varx)
            Sx = [j for j in J if relx[j] > .5]
            for i in I:
                c_Sx = sum([OperCostData(i, Sx, rawdata[k][i], rawdata[k][num_I:num_I+num_J], num_J, d)
                                for k in range(num_data)])/num_data

                print(i, Sx, c_Sx,relg[i])

                if relg[i] < c_Sx:
                    model.cbLazy(model._varg[i] >= c_Sx + quicksum((sum([OperCostData(i, Sx + [j], rawdata[k][i],
                                                                                     rawdata[k][num_I:num_I+num_J],
                                                                                     num_J, d) for k in range(num_data)])/num_data - c_Sx)
                                 * model._varx[j] for j in J if j not in Sx))

    model._varg = gamma
    model._varx = x
    model.Params.LazyConstraints = 1
    model.Params.Cuts = 0
    model.optimize(mycallback)

    tend = time.time()
    gap = model.MIPGap
    loc_chosen = [j for j in J if x[j].x > 0.5]


    obj_value = model.getObjective().getValue()

    return loc_chosen, obj_value, tend - tstart, gap
