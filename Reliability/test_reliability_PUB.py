import pandas as pd

from  Utility.Constants import NumSimulate, START_YEAR, END_YEAR
from Reliability.Reliability_PUB import RecoverEpsSamples,  GetOptEps_PUB_Ratio
from PUB.SUPDataDriven import SupLazy_AggragateCov
from SampleAverage.SAAFunc import SAACost
from DataRelated.DataProcess import PreProcess, SceProb
from datetime import date
import time
import numpy as np

def run_Kol_eps(rawdata_train_lst, rawdata_test, info, num_node, num_data, num_cov, beta_,eps):
    sim_data_driven_cost = np.zeros(NumSimulate)
    sim_out_of_sample_saa = np.zeros(NumSimulate)
    sim_in_sample_saa = np.zeros(NumSimulate)
    reliability = 0.0
    time_sol = 0.0
    avg_gap = 0.0
    outputfile_cov = open(
        'result/Reliability/Kol_cov_%s_%d_%d_%d_%f_EPS%f.csv' % (
        date.today().strftime("%Y_%m_%d"), num_data, num_node, num_cov, beta_, eps), 'w')

    outline0 = ['index', 'obj_value', 'in_sample_cost',  'out_of_sample_cost',  'StrLoc',  'sol_time', 'gap', 'rel']
    outputfile_cov.writelines(','.join(outline0) + '\n')
    for l in range(NumSimulate):
        # read the data
        rawdata_train = rawdata_train_lst[l]
        # rawdata_test = rawdata_test_lst[l]
        # info['num_cov'] = len(list(set(rawdata_train['cov'])))
        eps_samples = RecoverEpsSamples(info, rawdata_train, beta_, eps)
        # generate the decision using the optimal eps

        pre_data_process = PreProcess( rawdata_train, num_cov, num_node)
        delta_marginal_prob_cov, delta_zeta_cond, lambda_ = SceProb(rawdata_train, info['num_cov'], info['num_customer'],
                                                                    eps, eps_samples, pre_data_process)

        loc_kol, obj_kol, t_kol, gap_kol = SupLazy_AggragateCov(info,  delta_marginal_prob_cov, delta_zeta_cond, lambda_)
        # obj_kol2 = TotalCostCovTight(loc_kol, info, delta_marginal_prob_cov, delta, emp_prob_cond, eps_samples)
        # print(obj_kol, obj_kol2)
        saa_in = SAACost(loc_kol, info['fixed_cost'], info['dist'], num_node,info['max_demand'], rawdata_train)
        saa_kol = SAACost(loc_kol, info['fixed_cost'], info['dist'], num_node,info['max_demand'],  rawdata_test)
        # print("Kolmogorov with covarate-----------------------------------")
        # print(eps, eps_samples)
        # print(loc_kol)
        # print("obj %f, saa in %f, saa out %f"%(obj_kol, saa_in,saa_kol))


        time_sol += t_kol
        avg_gap += gap_kol
        str_loc = ";".join([str(x) for x in loc_kol])

        if saa_kol <= obj_kol:
            reliability += 1
        sim_data_driven_cost[l] = obj_kol
        # sim_in_sample_saa[l] = saa_in
        sim_out_of_sample_saa[l] = saa_kol
        outline_cov = ["%d" % l] + ["%.2f" % obj_kol] + ["%.2f" % saa_in] +  ["%.2f" % saa_kol] + ['%s' % str_loc] + ["%f" % t_kol] + ["%f"%gap_kol] + ['%d'%reliability]
        outputfile_cov.writelines(','.join(outline_cov) + '\n')

    # sim_data_driven_cost = sim_data_driven_cost / NumSimulate
    mean_in_sample_cost = np.average(sim_data_driven_cost)
    q1_in_sample_cost = np.quantile(sim_data_driven_cost, q = 0.1)
    q2_in_sample_cost = np.quantile(sim_data_driven_cost, q = 0.2)
    q8_in_sample_cost = np.quantile(sim_data_driven_cost, q = 0.8)
    q9_in_sample_cost = np.quantile(sim_data_driven_cost, q = 0.9)
    mean_in_sample_saa = np.average(sim_in_sample_saa)
    mean_out_of_sample_saa = np.average(sim_out_of_sample_saa)
    q1_out_of_sample_saa = np.quantile(sim_out_of_sample_saa, q = 0.1)
    q2_out_of_sample_saa = np.quantile(sim_out_of_sample_saa, q = 0.2)
    q8_out_of_sample_saa = np.quantile(sim_out_of_sample_saa, q = 0.8)
    q9_out_of_sample_saa = np.quantile(sim_out_of_sample_saa, q = 0.9)
    reliability = reliability / NumSimulate
    time_sol = time_sol / NumSimulate
    avg_gap = avg_gap / NumSimulate

    ## output reliability result
    outputfile = open(
        'result/Reliability_Kol_Node%d_Data%d_Cov%d_Beta%.4f_Eps%f.csv' % (num_node, num_data, num_cov, beta_,eps),
        'w')

    out_line_col0 = ['num_data', 'Method',   'in_sample_cost_avg',  'in_sample_cost_q1',
                     'in_sample_cost_q2', 'in_sample_cost_q8', 'in_sample_cost_q9',
                     'out_of_sample_cost_avg', 'out_of_sample_cost_q1',
                     'out_of_sample_cost_q2','out_of_sample_cost_q8','out_of_sample_cost_q9',
                     'out_of_sample_reliability', 'time_sol', 'Gap']
    outputfile.writelines(','.join(out_line_col0) + '\n')

    outline = ['%d' % num_data] + ['Kol_cov' ]  \
               + ['%.2f' % mean_in_sample_cost]  + ['%.2f' % q1_in_sample_cost] + ['%.2f' % q2_in_sample_cost]\
               + ['%.2f' % q8_in_sample_cost] + ['%.2f' % q9_in_sample_cost]\
               + ['%.2f' % mean_out_of_sample_saa] + ['%.2f' % q1_out_of_sample_saa] + ['%.2f' % q2_out_of_sample_saa] \
               + ['%.2f' % q8_out_of_sample_saa] + ['%.2f' % q9_out_of_sample_saa] \
               + ['%.4f' % reliability] + ["%f" % time_sol] + ['%.4f' % avg_gap]

    outputfile.writelines(','.join(outline) + '\n')
    outputfile_cov.close()
    outputfile.close()

def test_Reliability_PUB(rawdata_train_lst, rawdata_test, info, num_node, num_data, num_cov, beta_):

    Opt_EpsSet = 0
    Opt_EpsSamplesSet = [0 for k in range(num_cov)]

    sim_data_driven_cost = np.zeros(NumSimulate)
    sim_out_of_sample_saa = np.zeros(NumSimulate)
    # sim_in_sample_saa = np.zeros(NumSimulate)

    in_sample_rel = 0
    reliability = 0
    count_search = 0
    time_search = 0
    time_sol = 0
    avg_gap = 0
    t_search = 0

    outputfile_cov = open(
        'result/Reliability/Kol_cov_data%d_node%d_cov%d_beta%f.csv' % (num_data, num_node, num_cov,  beta_), 'w')



    outline0 = ['index', 'in_sample_cost',  'out_of_sample_cost', 'in_sample_rel', 'Str_loc', 'Eps']
    for k in range(num_cov):
        outline0 += ['Eps_samples_%d' %k]

    outline0 +=  ['Eps_Search_Time', 'Eps_all_time', 'num_search', 'sol_time', 'gap']
    outputfile_cov.writelines(','.join(outline0) + '\n')

    for l in range(NumSimulate):

        # read the data
        rawdata_train = rawdata_train_lst[l]
        # rawdata_test = rawdata_test_lst[l]

        pre_data_process = PreProcess(rawdata_train, num_cov, num_node)
        t1 = time.time()
        # search the optimal set of epsilon for kolmogorov method
        opt_eps, opt_eps_samples, opt_reliability, t_eps_cov, count_eps_cov = GetOptEps_PUB_Ratio(info, beta_, rawdata_train, pre_data_process)
        # opt_eps, opt_eps_samples, opt_reliability, t_eps_cov, count_eps_cov = GetOptEps_Kol_GridSearch(info, beta_, rawdata_train)
        t2 = time.time()
        # generate the decision using the optimal eps


        delta_marginal_prob_cov, delta_zeta_cond, lambda_ = SceProb(rawdata_train, info['num_cov'], info['num_customer'],
                                                                    opt_eps,
                                                           opt_eps_samples, pre_data_process)

        loc_kol, obj_kol, t_kol, gap_kol = SupLazy_AggragateCov(info, delta_marginal_prob_cov, delta_zeta_cond, lambda_)
        print(opt_eps, opt_eps_samples)
        # obj_kol2 = TotalCostCovTight(loc_kol,  info,  delta_marginal_prob_cov,  delta, emp_prob_cond, opt_eps_samples)

        # obj_kol = obj_kol2
        t_search += t_kol
        # calculate the out of sample cost
        saa_in = SAACost(loc_kol, info['fixed_cost'], info['dist'], num_node, info['max_demand'],rawdata_train)
        saa_kol = SAACost(loc_kol, info['fixed_cost'], info['dist'], num_node,info['max_demand'], rawdata_test)
        str_loc =  ";".join([str(x) for x in loc_kol])
        count_search  += count_eps_cov
        time_search  += t_eps_cov
        time_sol  += t_kol
        avg_gap  += gap_kol

        if saa_kol <= obj_kol:
            reliability += 1
        #
        # rel_kol = 0
        # for item in out_of_sample_lst_kol:
        #     if item < obj_kol:
        #         rel_kol += 1
        # opt_rel_kol_in = rel_kol / len(out_of_sample_lst_kol)
        opt_rel_kol_in = opt_reliability
        # record the details of each simulation in separete file
        outline_cov = ["%d" % l] +   ["%.2f" % obj_kol] +  ["%.2f" % saa_kol] + ["%.4f" % opt_rel_kol_in] \
                         + ['%s' % str_loc] + ["%f" % opt_eps]
        for k in range(len(opt_eps_samples)):
            outline_cov +=  ["%f" % opt_eps_samples[k]]
        for k in range(len(opt_eps_samples), num_cov):
            outline_cov += ['%f'% 0.0]

        outline_cov +=  ['%.2f' % t_eps_cov] + ['%.2f' % (t2 - t1)] + ['%d' % count_eps_cov] + ['%.2f' % t_kol] + ['%f' % gap_kol]
        outputfile_cov.writelines(','.join(outline_cov) + '\n')

        # record the parameter and results to the upper level group
        Opt_EpsSet += opt_eps

        for k in range(info['num_cov']):
            Opt_EpsSamplesSet[k] = Opt_EpsSamplesSet[k] + opt_eps_samples[k]


        sim_data_driven_cost[l]  = obj_kol
        sim_out_of_sample_saa[l] = saa_kol
        # sim_in_sample_saa[l] = saa_in
        in_sample_rel += opt_rel_kol_in


    Opt_EpsSet = Opt_EpsSet/NumSimulate
    Opt_EpsSamplesSet = [Opt_EpsSamplesSet[k]/NumSimulate for k in range(len(Opt_EpsSamplesSet))]
    # sim_data_driven_cost = sim_data_driven_cost / NumSimulate
    # sim_out_of_sample_saa[i] = sim_out_of_sample_saa[i] / NumSimulate
    mean_in_sample_cost = np.average(sim_data_driven_cost)
    q1_in_sample_cost = np.quantile(sim_data_driven_cost, q=0.1)
    q2_in_sample_cost = np.quantile(sim_data_driven_cost, q=0.2)
    q8_in_sample_cost = np.quantile(sim_data_driven_cost, q=0.8)
    q9_in_sample_cost = np.quantile(sim_data_driven_cost, q=0.9)

    # mean_in_sample_saa = np.average(sim_in_sample_saa)
    mean_out_of_sample_saa = np.average(sim_out_of_sample_saa)
    q1_out_of_sample_saa = np.quantile(sim_out_of_sample_saa, q = 0.1)
    q2_out_of_sample_saa = np.quantile(sim_out_of_sample_saa, q = 0.2)
    q8_out_of_sample_saa = np.quantile(sim_out_of_sample_saa, q = 0.8)
    q9_out_of_sample_saa = np.quantile(sim_out_of_sample_saa, q = 0.9)

    in_sample_rel = in_sample_rel / NumSimulate
    reliability = reliability / NumSimulate
    count_search = count_search/ NumSimulate
    time_search = time_search /NumSimulate
    time_sol= time_sol/NumSimulate
    avg_gap = avg_gap/NumSimulate
    ## output reliability result
    outputfile = open(
        'result/Reliability_Kol_Node%d_Data%d_Cov%d_Beta%.4f.csv' % (num_node, num_data, num_cov, beta_),
        'w')

    out_line_col0 = ['num_data', 'Method', 'count_search', 'time_search',
                      'in_sample_cost_avg', 'in_sample_cost_q1',
                     'in_sample_cost_q2', 'in_sample_cost_q8', 'in_sample_cost_q9',
                     'out_of_sample_cost_avg','out_of_sample_cost_q1','out_of_sample_cost_q2',
                     'out_of_sample_cost_q8','out_of_sample_cost_q9',
                     'in_sample_reliability',
                     'out_of_sample_reliability', 'Avg_eps', 'time_sol', 'Gap']
    outputfile.writelines(','.join(out_line_col0) + '\n')
    Method = 'Kol_cov'
    outline = ['%d' % num_data] + ['%s' % Method] + ['%d' % count_search] + ["%f" % time_search ] \
               + ['%.2f' % mean_in_sample_cost] + ['%.2f' % q1_in_sample_cost] + ['%.2f' % q2_in_sample_cost] \
               + ['%.2f' % q8_in_sample_cost] + ['%.2f' % q9_in_sample_cost]  + ['%.2f' % mean_out_of_sample_saa] \
               + ['%.2f' % q1_out_of_sample_saa] + ['%.2f' % q2_out_of_sample_saa] + ['%.2f' % q8_out_of_sample_saa] \
               + ['%.2f' % q9_out_of_sample_saa]  + ['%.4f' % in_sample_rel ] \
               + ['%.4f' % reliability] + ['%f' % Opt_EpsSet ]+ ["%f"%time_sol ]+['%.4f'%avg_gap ]


    outputfile.writelines(','.join(outline) + '\n')
    # for l in range(num_cov):
    #         outline += ['%f' % Opt_EpsSamplesSet[l]]
    outputfile_cov.close()
    outputfile.close()

    print("Reliability test of Kol Cov finished!")
    # return sim_data_driven_cost, sim_out_of_sample_saa, in_sample_rel, reliability, Opt_EpsSet, Opt_EpsSamplesSet, count_search, time_search




def run_Storm_PUB(train_data_lst, test_data_lst, info, num_node, num_cov, beta_,  train_length, test_length):
    outputfile = open(
        'result/Storm/Kol_cov-Node_%d-Cov_%d-Beta_%.2f.csv' % (num_node, num_cov,  beta_),
        'w')

    out_line_col0 = ['Method', 'Train_Start_year', 'Train_End_year', 'Test_Start_year', 'Test_End_year',
                      'obj', 'location', 'search_time', 'count_search', 'sol_time', 'OutofSample', 'Gap',
                     'OptEps', 'OptEpsSamples']
    outputfile.writelines(','.join(out_line_col0) + '\n')
    # print(len(train_data_lst))


    for t in range(START_YEAR, END_YEAR + 1 - train_length):
        rawdata_train = train_data_lst[t-START_YEAR]
        rawdata_test = test_data_lst[t-START_YEAR]
        print(t, len(rawdata_train))
        info['num_cov'] = len(list(set(rawdata_train['cov'])))
        # # generate the train and test data
        # rawdata_train, rawdata_test, info_train, info_test = RawDataGenerate_train_test(num_node, num_data, num_cov)

        # the kolomogorov with covariate

        pre_data_process = PreProcess(rawdata_train, num_cov, num_node)

        # search the optimal set of epsilon for kolmogorov method
        opt_eps, opt_eps_samples, opt_reliability, t_eps_cov, count_eps_cov = GetOptEps_PUB_Ratio(
            info, beta_, rawdata_train, pre_data_process)
        print(opt_eps, opt_eps_samples)
        # generate the decision using the optimal eps
        delta_marginal_prob_cov, delta_zeta_cond, lambda_ = SceProb(rawdata_train, info['num_cov'],
                                                                   info['num_customer'],
                                                                    opt_eps,
                                                                    opt_eps_samples, pre_data_process)

        loc_kol, obj_kol, t_kol, gap_kol = SupLazy_AggragateCov(info, delta_marginal_prob_cov, delta_zeta_cond, lambda_)

        str_loc = ";".join([str(x) for x in loc_kol])

        # calculate the out of sample cost
        saa_kol = SAACost(loc_kol, info['fixed_cost'], info['dist'], num_node,info['max_demand'],rawdata_test)


        #
        # outputfile.close()

        outline1 =['%s' % 'Kol_cov'] + ['%d' % (t)] + ['%d' % (t+train_length-1)] \
                   +['%d' % (t+train_length)] + ['%d' % (t+train_length+test_length-1)]  \
                   + ['%.2f' % obj_kol] + ['%s'%str_loc] + ['%.6f' % t_eps_cov] + ['%d'%count_eps_cov] + ['%.6f' % t_kol]  \
                  + ['%.2f' % saa_kol]  +['%.4f'% gap_kol] + ['%f' % opt_eps]
        for l in range(len(opt_eps_samples)):
            outline1 += ['%.4f' % opt_eps_samples[l]]
        outputfile.writelines(','.join(outline1) + '\n')


    outputfile.close()
    print("Storm Case of Kol Cov finished!")