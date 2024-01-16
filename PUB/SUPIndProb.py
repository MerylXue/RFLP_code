# """
# Code for "Hao Shen, Mengying Xue, and Zuojun Max Shen (2023),
# Data-driven Reliable Faciltiy Location Problem, Management Science, Forthcoming",
#
# Last Revised on Jan 12, 2024
# @author: Mengying Xue
# """
#
# from gurobipy import *
# import time
#
# def SupLazy_IndProb(info, IndProb):
#     num_I = info['num_customer']
#     num_J = info['num_facility']
#     mu = info['mu']
#     f = info['fixed_cost']
#     dist = info['dist']
#     I = J = range(num_I)
#     tstart = time.time()
#
#     model = Model('SUP_IndProb')
#     model.modelSense = GRB.MINIMIZE
#     x = model.addVars(num_J, vtype=GRB.BINARY, obj=f[:-1])
#     gamma = model.addVars(num_I, obj=1)
#
#     # model.addConstr(quicksum(x[j] for j in J) <= P)
#     model.addConstr(quicksum(x[j] for j in J) >= 1)
#     model.addConstrs(
#         (gamma[i] >= OperCostIndProb(i, [j for j in J], IndProb, mu[i], dist)
#          for i in I)
#     )
#     count = 0
#     def mycallback(model, where):
#         if where == GRB.Callback.MIPSOL:
#             relg = model.cbGetSolution(model._varg)
#             relx = model.cbGetSolution(model._varx)
#             Sx = [j for j in J if relx[j] > .5]
#
#             for i in I:
#                 c_Sx = OperCostIndProb(i, Sx,
#                                 IndProb, mu[i], dist)
#                 if relg[i] < c_Sx:
#                     model.cbLazy(model._varg[i] >= c_Sx \
#                                  + LinExpr([OperCostIndProb(i, Sx + [j],IndProb, mu[i], dist) - c_Sx
#                                             for j in J if j not in Sx],
#                                            [model._varx[j] for j in J if j not in Sx]))
#
#
#     model._varg = gamma
#     model._varx = x
#     model.Params.LazyConstraints = 1
#     model.Params.Cuts = 0
#     model.optimize(mycallback)
#     # model.write("ind.lp")
#
#     obj_val = model.getObjective()
#
#     tend = time.time()
#     loc_chosen = [j for j in J if x[j].x > 0.5]
#     # gamma_sol = [gamma[j].x for j in J]
#     #
#     return loc_chosen, obj_val.getValue(), tend-tstart, model.MIPGap
