from gurobipy import *
import time
from MomentBased.CrossMomentFunc import OperCostCrossMomentCov, OperCostCrossMomentNoCov


def SupLazy_SecondCrossCov(info, MeanDemand, delta_marginal_prob_cov, MarginalProb, SecondMomentProb, IndexPair):
    f = info['fixed_cost']

    num_I = info['num_customer']
    num_J = info['num_facility']
    dist = info['dist']
    num_cov = info['num_cov']
    I = J = range(num_I)
    tstart = time.time()

    model = Model('SUP')
    model.Params.OutputFlag = False
    model.setParam(GRB.Param.TimeLimit, 900)

    model.modelSense = GRB.MINIMIZE
    x = model.addVars(num_J, vtype=GRB.BINARY, obj=f[:-1])
    gamma = model.addVars(num_I, obj=1)

    # model.addConstr(quicksum(x[j] for j in J) >= 1)

    model.addConstrs(
        (gamma[i] >= OperCostCrossMomentCov(i, [j for j in range(num_J)], num_cov,
                                         delta_marginal_prob_cov, MarginalProb, SecondMomentProb, IndexPair, MeanDemand[:,i], dist)
         for i in I)
    )

    def mycallback(model, where):
        if where == GRB.Callback.MIPSOL:
            relg = model.cbGetSolution(model._varg)
            relx = model.cbGetSolution(model._varx)
            Sx = [j for j in J if relx[j] > .5]

            for i in I:
                c_Sx = OperCostCrossMomentCov(i, Sx, num_cov,
                                         delta_marginal_prob_cov,MarginalProb, SecondMomentProb, IndexPair, MeanDemand[:,i], dist)
                if relg[i] < c_Sx:
                    model.cbLazy(model._varg[i] >= c_Sx
                                 + quicksum((OperCostCrossMomentCov(i, Sx + [j],num_cov,
                                         delta_marginal_prob_cov,MarginalProb, SecondMomentProb, IndexPair, MeanDemand[:,i], dist)
                                             - c_Sx) * model._varx[j] for j in J if j not in Sx))
                                 # + LinExpr([OperCostCrossMomentCov(i, Sx + [j],num_cov,
                                 #         delta_marginal_prob_cov,MarginalProb, SecondMomentProb, IndexPair, MeanDemand[:,i], dist) - c_Sx
                                 #            for j in J if j not in Sx],
                                 #           [model._varx[j] for j in J if j not in Sx]))

    model._varg = gamma
    model._varx = x
    model.Params.LazyConstraints = 1
    model.Params.Cuts = 0
    model.optimize(mycallback)

    x_sol = [x[i].x for i in range(num_J )]
    # print(x_sol)
    obj_val = model.getObjective()

    chosen_loc = [j for j in range(num_J ) if x_sol[j] > 0.5]
    # print(chosen_loc)
    tend = time.time()
    # print('SupLazy takes %.2fs' % (tend - tstart))
    return chosen_loc, obj_val.getValue(), tend - tstart, model.MIPGap



def SupLazy_SecondCrossNoCov(info, MeanDemand,MarginalProb, SecondMomentProb, IndexPair):
    f = info['fixed_cost']

    num_I = info['num_customer']
    num_J = info['num_facility']
    dist = info['dist']
    I = J = range(num_I)
    tstart = time.time()

    model = Model('SUP')
    model.modelSense = GRB.MINIMIZE
    model.Params.OutputFlag = False
    model.setParam(GRB.Param.TimeLimit, 900)

    x = model.addVars(num_J, vtype=GRB.BINARY, obj=f[:-1])
    gamma = model.addVars(num_I, obj=1)

    # model.addConstr(quicksum(x[j] for j in J) >= 1)
    model.addConstrs(
        (gamma[i] >= OperCostCrossMomentNoCov(i, [j for j in range(num_J)], MarginalProb, SecondMomentProb, IndexPair, MeanDemand[i], dist)
         for i in I)
    )

    def mycallback(model, where):
        if where == GRB.Callback.MIPSOL:
            relg = model.cbGetSolution(model._varg)
            relx = model.cbGetSolution(model._varx)
            Sx = [j for j in J if relx[j] > .5]

            for i in I:
                c_Sx = OperCostCrossMomentNoCov(i, Sx, MarginalProb, SecondMomentProb, IndexPair, MeanDemand[i], dist)
                if relg[i] < c_Sx:
                    model.cbLazy(model._varg[i] >= c_Sx
                                 + quicksum((OperCostCrossMomentNoCov(i, Sx + [j],MarginalProb, SecondMomentProb, IndexPair, MeanDemand[i], dist)
                                             - c_Sx) * model._varx[j] for j in J if j not in Sx))
                                 # + LinExpr([OperCostCrossMomentNoCov(i, Sx + [j],MarginalProb, SecondMomentProb, IndexPair, MeanDemand[i], dist) - c_Sx
                                 #            for j in J if j not in Sx],
                                 #           [model._varx[j] for j in J if j not in Sx]))

    model._varg = gamma
    model._varx = x
    model.Params.LazyConstraints = 1
    model.Params.Cuts = 0
    model.optimize(mycallback)

    x_sol = [x[i].x for i in range(num_J )]
    # print(x_sol)
    obj_val = model.getObjective()

    chosen_loc = [j for j in range(num_J ) if x_sol[j] > 0.5]
    # print(chosen_loc)
    tend = time.time()
    # print('SupLazy takes %.2fs' % (tend - tstart))
    return chosen_loc, obj_val.getValue(), tend - tstart, model.MIPGap