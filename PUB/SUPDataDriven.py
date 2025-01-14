from gurobipy import *
import time
from PUB.SUPFunc import  TotalFixedCost, OperCostCov, TotalOperCostCov



def SupLazy_AggragateCov(info,  delta_marginal_prob_cov, delta_zeta_cond, lambda_):
    num_I = info['num_customer']
    num_J = info['num_facility']
    I = range(num_I)
    J = range(num_J)
    d = info['dist']
    num_K = info['num_cov']

    f = info['fixed_cost']
    max_demand = info['max_demand']
    tstart = time.time()
    wc_d = [{} for i in I]

    model = Model('SUP')
    model.modelSense = GRB.MINIMIZE
    # model.Params.OutputFlag = False
    model.setParam(GRB.Param.TimeLimit, 900)

    x = model.addVars(num_J, vtype=GRB.BINARY, obj=f[:-1])
    gamma = model.addVars(num_I, obj=[1 for i in I])
    ini_sol = [j for j in J]
    # model.addConstr(quicksum(x[j] for j in J) >= 1)
    model.addConstrs(
        (gamma[i] >= OperCostCov(i, ini_sol, num_K, delta_marginal_prob_cov,delta_zeta_cond[i], lambda_[i], num_J, d, wc_d[i])
         for i in I )
    )

    def mycallback(model, where):
        if where == GRB.Callback.MIPSOL:
            relg = model.cbGetSolution(model._varg)
            relx = model.cbGetSolution(model._varx)
            Sx = [j for j in J if relx[j] > .5]
            # print(Sx)
            # print(relg)
            # print(relx)

            for i in I:
                c_Sx = OperCostCov(i, Sx, num_K, delta_marginal_prob_cov,
                                        delta_zeta_cond[i],lambda_[i], num_J, d, wc_d[i])
                # print(i,c_Sx,relg[i])

                if relg[i] < c_Sx:
                    # print([OperCostCov(i, Sx + [j], num_K, delta_marginal_prob_cov, delta_zeta_cond[i],
                    #                                     lambda_[i],num_J, d, wc_d[i]) - c_Sx for j in J if j not in Sx])
                    # model.cbLazy(model._varg[i] >= c_Sx \
                    #              + LinExpr([OperCostCov(i, Sx + [j], num_K, delta_marginal_prob_cov, delta_zeta_cond[i],
                    #                                     lambda_[i],num_J, d, wc_d[i]) - c_Sx for j in J if j not in Sx],
                    #                        [model._varx[j] for j in J if j not in Sx]))
                    model.cbLazy(model._varg[i] >= c_Sx \
                                 + quicksum((OperCostCov(i, Sx + [j], num_K, delta_marginal_prob_cov, delta_zeta_cond[i],
                                                        lambda_[i], num_J, d, wc_d[i]) - c_Sx) * model._varx[j]
                                            for j in J if j not in Sx))

    model._varg = gamma
    model._varx = x
    model.Params.LazyConstraints = 1
    model.Params.Cuts = 0
    model.optimize(mycallback)

    tend = time.time()
    gap = model.MIPGap
    loc_chosen = [j for j in J if x[j].x > 0.5]

    # for i in range(num_I):
    #     cost = OperCostCov(i, loc_chosen, num_K, delta_marginal_prob_cov, delta_zeta_cond[i],
    #                                                     lambda_[i],num_J, d, wc_d[i])
    #     gamma_ = gamma[i].x
    #     # print(cost, gamma_)
    #
    #
    # total_oper_cost = TotalOperCostCov(loc_chosen, num_I, num_K, delta_marginal_prob_cov, delta_zeta_cond, lambda_, num_J, d,  wc_d)
    # total_fixed_cost = TotalFixedCost(loc_chosen, f)
    obj_value = model.getObjective().getValue()

    # print("obj %f, sum_lambda %f"%(obj_value,
    #                                sum([delta_marginal_prob_cov[k] * d[i][-1] * lambda_[i][k] for i in I for k in range(num_K)])))
    return loc_chosen, obj_value, tend - tstart, gap


