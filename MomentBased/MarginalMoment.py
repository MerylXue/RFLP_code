
import numpy as np
from gurobipy import *
import time

def SequencedScenarios(MarginalProb):
    num_J = num_I = len(MarginalProb)
    MaringalProb = list(MarginalProb) + [0]

    temp = [(j, MaringalProb[j]) for j in range(num_J+1)]
    temp_sort = sorted(temp, key=lambda a: a[1], reverse = False)#ascending order
    # print(temp_sort)
    Scenarios = [[] for j in range(num_J+1)]
    for j in range(num_J+1):
        Scenarios[j] = [1 for j in range(num_J+1)]
        for l in range(j+1,num_J+1):
            Scenarios[j][temp_sort[l][0]] = 0

    return Scenarios, [a[0] for a in temp_sort]


def SolveRobustCov(info, MeanDemand, Scenario_prob,  MarginalProbCov):
    # demand_val = info['mu']
    f = info['fixed_cost']
    start_time = time.time()
    num_J = info['num_facility']
    num_I = info['num_customer']
    dist = info['dist']
    # print(sum(Scenario_prob))
    # print(MarginalProbCov)
    ScenarioLst = []
    SortedLocLst = []
    num_cov = len(Scenario_prob )
    for k in range(num_cov):
        Scenarios, SortedLoc = SequencedScenarios(MarginalProbCov[k])
        ScenarioLst.append(Scenarios)
        SortedLocLst.append(SortedLoc)
        MarginalProbCov[k] = MarginalProbCov[k]+[0]
    # print(MarginalProbCov)
    # print(ScenarioLst)
    # print(SortedLocLst)
    model = Model('RP_Cov')
    model.modelSense = GRB.MINIMIZE
    model.Params.OutputFlag = False
    x = model.addVars(num_J + 1, vtype=GRB.BINARY)
    y = model.addVars(num_cov, num_I, num_J + 1, num_J + 1, vtype=GRB.BINARY)
    # obj = LinExpr(f[:-1], [x[j] for j in range(num_J)])
    obj = quicksum(f[j] * x[j] for j in range(num_J))
    for k in range(num_cov):
        for i in range(num_I):
            for j in range(num_J+1):
                for l in range(num_J):
                    obj.add(dist[i][j] * MeanDemand[k][i] * Scenario_prob[k] *
                            (MarginalProbCov[k][SortedLocLst[k][l+1]] - MarginalProbCov[k][SortedLocLst[k][l]]) * y[k,i,j,l])
                obj.add(dist[i][j] * MeanDemand[k][i]  * Scenario_prob[k] *
                        (1 - MarginalProbCov[k][SortedLocLst[k][num_J]]) * y[k,i,j,num_J])
    # obj.add(quicksum(dist[i][j] * MeanDemand[k][i] * Scenario_prob[k] * (MarginalProbCov[k][SortedLocLst[k][l+1]]
    #                                                                      - MarginalProbCov[k][SortedLocLst[k][l]]) * y[k,i,j,l]
    #                  for k in range(num_cov) for i in range(num_I) for j in range(num_J) for l in range(num_J)))
    # obj.add(quicksum(dist[i][j] * MeanDemand[k][i] * Scenario_prob[k] *
    #                             (1 - MarginalProbCov[k][SortedLocLst[k][num_J]]) * y[k,i,j,num_J]
    #                  for k in range(num_cov)
    #                  for i in range(num_I)
    #                  for j in range(num_J)))
    # model.addConstr(quicksum(x[j] for j in range(num_J)) >= 1)
    model.addConstr(x[num_J] == 1)
    model.addConstrs(
        (quicksum(y[k,i, j, l] for j in range(num_J + 1)) == 1 for i in range(num_I) for l in range(num_J + 1) for k in range(num_cov)))

    model.addConstrs((y[k,i, j, l] <= x[j] * ScenarioLst[k][l][j] for i in range(num_I) for j in range(num_J) for l in
                      range(num_J + 1) for k in range(num_cov)))

    model.setParam(GRB.Param.TimeLimit, 900)
    model.setObjective(obj, GRB.MINIMIZE)
    model.optimize()
    x_sol = [x[j].x for j in range(num_J+1)]
    # print(x_sol)
    location_index = [j for j in range(num_J) if x_sol[j] == 1]
    cost = model.getObjective().getValue()
    end_time = time.time()
    # print("Robust solution time %f" %(end_time - start_time))
    return location_index, cost, end_time - start_time, model.MIPGap



def SolveRobust(info, MeanDemand, MarginalProb):
    f = info['fixed_cost']
    start_time = time.time()
    num_J = info['num_facility']
    num_I = info['num_customer']
    dist = info['dist']
    Scenarios, SortedLoc = SequencedScenarios(MarginalProb)
    MarginalProb = MarginalProb + [0]

    model = Model('RP')
    model.modelSense = GRB.MINIMIZE
    model.Params.OutputFlag = False
    x = model.addVars(num_J + 1, vtype=GRB.BINARY)
    y = model.addVars(num_I, num_J + 1, num_J + 1, vtype= GRB.BINARY)


    # obj = LinExpr(f, [x[j] for j in range(num_J+1)])
    obj = quicksum(f[j] * x[j] for j in range(num_J))

    for i in range(num_I):
        for j in range(num_J+1):
            for l in range(num_J):
                obj.add(dist[i][j] * MeanDemand[i] * (MarginalProb[SortedLoc[l+1]] - MarginalProb[SortedLoc[l]]) * y[i,j,l])
            obj.add(dist[i][j] * MeanDemand[i] * (1 - MarginalProb[SortedLoc[num_J]]) * y[i,j,num_J])
    # obj.add(quicksum(dist[i][j] * MeanDemand[i] * (MarginalProb[SortedLoc[l + 1]] - MarginalProb[SortedLoc[l]]) * y[i, j, l]
    #                  for i in range(num_I) for j in range(num_J) for l in range(num_J)))
    # obj.add(quicksum(dist[i][j] * MeanDemand[i] * (1 - MarginalProb[SortedLoc[num_J]]) * y[i, j, num_J]
    #                  for i in range(num_I) for j in range(num_J)))

    # model.addConstr(quicksum(x[j] for j in range(num_J)) >= 1)
    model.addConstr(x[num_J] == 1)
    model.addConstrs((quicksum(y[i,j,l] for j in range(num_J+1)) == 1 for i in range(num_I) for l in range(num_J+1)))
    model.addConstrs((y[i,j,l] <= x[j] * Scenarios[l][j] for i in range(num_I) for j in range(num_J) for l in range(num_J+1)))
    model.setObjective(obj, GRB.MINIMIZE)

    model.setParam(GRB.Param.TimeLimit, 900)
    model.optimize()
    x_sol = [x[j].x for j in range(num_J+1)]
    # print(x_sol)
    location_index = [j for j in range(num_J) if x_sol[j] == 1]
    cost = model.getObjective().getValue()
    end_time = time.time()
    # print("Robust solution time %f" %(end_time - start_time))
    return location_index, cost, end_time - start_time, model.MIPGap