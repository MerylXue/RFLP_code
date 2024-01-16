"""
Code for "Hao Shen, Mengying Xue, and Zuojun Max Shen (2023),
Data-driven Reliable Faciltiy Location Problem, Management Science, Forthcoming",

Last Revised on Jan 12, 2024
@author: Mengying Xue

"""

from  Utility.Constants import NumSimulate, start_year, end_year
from SampleAverage.SAAFunc import SAACost
from datetime import date
import numpy as np
from MomentBased.SUPCrossMoment import SupLazy_SecondCrossNoCov, SupLazy_SecondCrossCov
from MomentBased.MarginalMoment import SolveRobust, SolveRobustCov
from MomentBased.MomentData import GenerateMomentDataCov, GenerateMomentDataNoCov


## return the reliability, average optimization performance for
# marginal and cross moment methods in both with covariates and no covariate case
# data: synthetic data
def test_Reliability_moment(rawdata_train_lst, rawdata_test, info, num_node, num_data, num_cov):
    n_method = 4
    ## remark the position of each method in the output file
    idx_cm_cov = 0
    idx_cm_novov = 1
    idx_mm_cov = 2
    idx_mm_nocov = 3

    sim_data_driven_cost= np.zeros((n_method, NumSimulate))
    mean_in_sample_cost = [0 for i in range(n_method)]
    q1_in_sample_cost = [0 for i in range(n_method)]
    q2_in_sample_cost = [0 for i in range(n_method)]
    q8_in_sample_cost = [0 for i in range(n_method)]
    q9_in_sample_cost = [0 for i in range(n_method)]

    sim_out_of_sample_saa_lst = np.zeros((n_method, NumSimulate))
    mean_out_of_sample_saa = [0 for i in range(n_method)]
    q1_out_of_sample_saa = [0 for i in range(n_method)]
    q2_out_of_sample_saa = [0 for i in range(n_method)]
    q8_out_of_sample_saa = [0 for i in range(n_method)]
    q9_out_of_sample_saa = [0 for i in range(n_method)]
    reliability = [0.0 for i in range(n_method)]
    time_sol = [0.0 for i in range(n_method)]
    gap_lst = [0.0 for i in range(n_method)]



    # output the results for four methods in four separate files
    outputfile_mmc = open(
        'result/Reliability/CrossMoment_cov_data%d_node%d_cov%d.csv' % (
             num_data, num_node, num_cov), 'w')

    outputfile_mmn = open(
        'result/Reliability/CrossMoment_nocov_data%d_node%d_cov%d.csv' % (
            num_data, num_node, num_cov), 'w')

    outputfile_lvc = open('result/Reliability/MarginalMoment_cov_data%d_node%d_cov%d.csv' % (
        num_data, num_node, num_cov), 'w')
    outputfile_lvn = open('result/Reliability/MarginalMoment_nocov_data%d_node%d_cov%d.csv' % (
             num_data, num_node, num_cov), 'w')
    outline1 = ['index', 'in_sample_cost', 'out_of_sample_cost', 'Str_loc', 'sol_time', 'gap']

    outputfile_mmc.writelines(','.join(outline1) + '\n')
    outputfile_mmn.writelines(','.join(outline1) + '\n')
    outputfile_lvc.writelines(','.join(outline1) + '\n')
    outputfile_lvn.writelines(','.join(outline1) + '\n')
    for l in range(NumSimulate):
        # read the data
        rawdata_train = rawdata_train_lst[l]

        mean_demand_mmc, marginal_prob_mmc, marginal_prob_disrupt_mmc, SecondMomentProb_mmc, IndexPair_mmc = GenerateMomentDataCov(info,
            rawdata_train, num_cov,
            info['num_customer'],
            info['num_facility'])
        loc_chosen_mmc, obj_value_mmc, t_mmc, gap_mmc = SupLazy_SecondCrossCov(info, mean_demand_mmc, marginal_prob_mmc,
                                                                               marginal_prob_disrupt_mmc,
                                                                               SecondMomentProb_mmc,
                                                                               IndexPair_mmc)
        str_loc_chosen_mmc = ";".join([str(x) for x in loc_chosen_mmc])


        saa_mmc = SAACost(loc_chosen_mmc, info['fixed_cost'], info['dist'], info['num_customer'], info['max_demand'], rawdata_test)
        sim_data_driven_cost[idx_cm_cov][l] = obj_value_mmc
        sim_out_of_sample_saa_lst[idx_cm_cov][l] = saa_mmc
        time_sol[idx_cm_cov] += t_mmc
        gap_lst[idx_cm_cov] += gap_mmc
        if saa_mmc <= obj_value_mmc:
            reliability[idx_cm_cov] += 1
        # record the details of each simulation in separete file
        outline_mmc = ["%d" % l] + ["%.2f" % obj_value_mmc] + ["%.2f" % saa_mmc] \
                      + ['%s' % str_loc_chosen_mmc] + ['%.2f' % t_mmc] + ['%f' % gap_mmc]
        outputfile_mmc.writelines(','.join(outline_mmc) + '\n')


        ## second cross moment with no covariate information
        mean_demand_mmn, marginal_prob_disrupt_mmn, SecondMomentProb_mmn, IndexPair_mmn = GenerateMomentDataNoCov(info,
            rawdata_train, info[
                'num_customer'], info['num_facility'], )
        loc_chosen_mmn, obj_value_mmn, t_mmn, gap_mmn = SupLazy_SecondCrossNoCov(info, mean_demand_mmn,
                                                                                 marginal_prob_disrupt_mmn,
                                                                                 SecondMomentProb_mmn, IndexPair_mmn)

        str_loc_chosen_mmn = ";".join([str(x) for x in loc_chosen_mmn])
        # print(loc_chosen_mmn, obj_value_mmn, t_mmn, gap_mmn)
        saa_mmn = SAACost(loc_chosen_mmn, info['fixed_cost'], info['dist'], info['num_customer'], info['max_demand'], rawdata_test)

        sim_data_driven_cost[idx_cm_novov][l] = obj_value_mmn
        sim_out_of_sample_saa_lst[idx_cm_novov][l] = saa_mmn
        time_sol[idx_cm_novov] += t_mmn
        gap_lst[idx_cm_novov] += gap_mmn

        if saa_mmn <= obj_value_mmn:
            reliability[idx_cm_novov] += 1

        # record the details of each simulation in separete file
        outline_mmn = ["%d" % l] + ["%.2f" % obj_value_mmn] + ["%.2f" % saa_mmn] \
                      + ['%s' % str_loc_chosen_mmn] + ['%.2f' % t_mmn] + ['%f' % gap_mmn]
        outputfile_mmn.writelines(','.join(outline_mmn) + '\n')
        ##marginal moment method with covariate in Lv 2015
        loc_lvc, obj_lvc, t_lvc, gap_lvc = SolveRobustCov(info, mean_demand_mmc, marginal_prob_mmc, marginal_prob_disrupt_mmc)
        saa_lvc = SAACost(loc_lvc, info['fixed_cost'], info['dist'], info['num_customer'], info['max_demand'], rawdata_test)
        str_loc_lvc = ";".join([str(x) for x in loc_lvc])

        sim_data_driven_cost[idx_mm_cov][l] = obj_lvc
        sim_out_of_sample_saa_lst[idx_mm_cov][l] = saa_lvc
        time_sol[idx_mm_cov] += t_lvc
        gap_lst[idx_mm_cov] += gap_lvc

        if saa_lvc <= obj_lvc:
            reliability[idx_mm_cov] += 1

        # record the details of each simulation in separete file
        outline_lvc = ["%d" % l] + ["%.2f" % obj_lvc] + ["%.2f" % saa_lvc] \
                      + ['%s' % str_loc_lvc] + ['%.2f' % t_lvc] + ['%f' % gap_lvc]
        outputfile_lvc.writelines(','.join(outline_lvc) + '\n')


        ## Marginal moment robust with no covariate in lv
        loc_lvn, obj_lvn, t_lvn, gap_lvn = SolveRobust(info, mean_demand_mmn, marginal_prob_disrupt_mmn)
        saa_lvn = SAACost(loc_lvn, info['fixed_cost'], info['dist'], info['num_customer'], info['max_demand'], rawdata_test)
        str_loc_lvn = ";".join([str(x) for x in loc_lvn])
        sim_data_driven_cost[idx_mm_nocov][l] = obj_lvn
        sim_out_of_sample_saa_lst[idx_mm_nocov][l] = saa_lvn
        time_sol[idx_mm_nocov] += t_lvn
        gap_lst[idx_mm_nocov] += gap_lvn

        if saa_lvn <= obj_lvn:
            reliability[idx_mm_nocov] += 1

        # record the details of each simulation in separete file
        outline_lvn = ["%d" % l] + ["%.2f" % obj_lvn] + ["%.2f" % saa_lvn] \
                      + ['%s' % str_loc_lvn] + ['%.2f' % t_lvn] + ['%f' % gap_lvn]
        outputfile_lvn.writelines(','.join(outline_lvn) + '\n')


    for i in range(n_method):
        mean_in_sample_cost[i] = np.average(sim_data_driven_cost[i])
        q1_in_sample_cost[i] = np.quantile(sim_data_driven_cost[i], q=0.1)
        q2_in_sample_cost[i] = np.quantile(sim_data_driven_cost[i], q=0.2)
        q8_in_sample_cost[i] = np.quantile(sim_data_driven_cost[i], q=0.8)
        q9_in_sample_cost[i] = np.quantile(sim_data_driven_cost[i], q=0.9)

        reliability[i] = reliability[i] / NumSimulate
        mean_out_of_sample_saa[i] = np.average(sim_out_of_sample_saa_lst[i])
        q1_out_of_sample_saa[i] = np.quantile(sim_out_of_sample_saa_lst[i], q = 0.1)
        q2_out_of_sample_saa[i] = np.quantile(sim_out_of_sample_saa_lst[i], q = 0.2)
        q8_out_of_sample_saa[i] = np.quantile(sim_out_of_sample_saa_lst[i], q = 0.8)
        q9_out_of_sample_saa[i] = np.quantile(sim_out_of_sample_saa_lst[i], q = 0.9)
        time_sol[i] = time_sol[i] / NumSimulate
        gap_lst[i] = gap_lst[i] / NumSimulate


    ## output reliability result
    outputfile = open(
        'result/Reliability_moment_Node%d_Data%d_Cov%d.csv' % (num_node, num_data, num_cov),
        'w')
    out_line_col0 = ['num_data', 'Method', 'in_sample_cost_avg', 'in_sample_cost_q1',
                     'in_sample_cost_q2', 'in_sample_cost_q8', 'in_sample_cost_q9',
                     'out_of_sample_cost_avg',
                     'out_of_sample_cost_q1', 'out_of_sample_cost_q2',
                     'out_of_sample_cost_q8', 'out_of_sample_cost_q9',
                     'out_of_sample_reliability', 'time_sol', 'Avg_gap']
    outputfile.writelines(','.join(out_line_col0) + '\n')
    Method = ['CM_cov', 'CM_nocv', 'MM_cov', 'MM_nocov']
    for i in range(len(Method)):
        outline = ['%d' % num_data]
        outline += ['%s' % Method[i]] + ['%.2f' % mean_in_sample_cost[i]] + ['%.2f' % q1_in_sample_cost[i]] \
                   + ['%.2f' % q2_in_sample_cost[i]] + ['%.2f' % q8_in_sample_cost[i]] + ['%.2f' % q9_in_sample_cost[i]]\
                   + ['%.2f' % mean_out_of_sample_saa[i]] + ['%.2f' % q1_out_of_sample_saa[i]]\
                   + ['%.2f' % q2_out_of_sample_saa[i]] + ['%.2f' % q8_out_of_sample_saa[i]] \
                   + ['%.2f' % q9_out_of_sample_saa[i]]  + ['%.4f' % reliability[i]] + ["%.4f" % time_sol[i]] + [
                       "%.4f" % gap_lst[i]]

        outputfile.writelines(','.join(outline) + '\n')

    outputfile.close()

    print("Reliability test of moment based method finished!")
    # return sim_data_driven_cost, sim_out_of_sample_saa, in_sample_rel, reliability, Opt_EpsSet, Opt_EpsSamplesSet, count_search, time_search



## return the reliability, average optimization performance for
# marginal and cross moment methods in both with covariates and no covariate case
# data: storm data case
# output the results of four methods in one file
def run_Storm_moment(train_data_lst, test_data_lst, info, num_node, num_cov, beta_, train_length, test_length):
    outputfile = open(
        'result/Storm/Moment-Date_%s-Node_%d-Cov_%d-Beta_%.2f.csv' % (date.today().strftime("%Y_%m_%d"), num_node, num_cov,  beta_),
        'w')

    out_line_col0 = ['Method', 'Train_Start_year', 'Train_End_year', 'Test_Start_year', 'Test_End_year',
                      'obj', 'location',  'sol_time', 'OutofSample', 'Gap']
    outputfile.writelines(','.join(out_line_col0) + '\n')
    # print(len(train_data_lst))

    for t in range(start_year, end_year + 1 - train_length):
        rawdata_train = train_data_lst[t-start_year]
        rawdata_test = test_data_lst[t-start_year]

        info['num_cov'] = len(list(set(rawdata_train['cov'])))
        # # generate the train and test data
        # rawdata_train, rawdata_test, info_train, info_test = RawDataGenerate_train_test(num_node, num_data, num_cov)

        # the kolomogorov with covariate

        mean_demand_mmc, marginal_prob_mmc, marginal_prob_disrupt_mmc, SecondMomentProb_mmc, IndexPair_mmc = GenerateMomentDataCov(info,
            rawdata_train, num_cov,
            info['num_customer'],
            info['num_facility'])
        loc_chosen_mmc, obj_value_mmc, t_mmc, gap_mmc = SupLazy_SecondCrossCov(info, mean_demand_mmc, marginal_prob_mmc,
                                                                               marginal_prob_disrupt_mmc,
                                                                               SecondMomentProb_mmc,
                                                                               IndexPair_mmc)
        # print(loc_chosen_mmc, obj_value_mmc, t_mmc, gap_mmc)
        saa_mmc = SAACost(loc_chosen_mmc, info['fixed_cost'], info['dist'], info['num_customer'], info['max_demand'],rawdata_test)

        str_mmc = ";".join([str(x) for x in loc_chosen_mmc])


        outline1 =['%s' % 'CM_cov'] + ['%d' % (t)] + ['%d' % (t+train_length-1)] \
                   +['%d' % (t+train_length)] + ['%d' % (t+train_length+test_length-1)]  \
                   + ['%.2f' % obj_value_mmc] + ['%s'%str_mmc] + ['%.6f' % t_mmc]  \
                  + ['%.2f' % saa_mmc]  +['%.4f'% gap_mmc]
        outputfile.writelines(','.join(outline1) + '\n')

        ## second cross moment with no covariate information
        mean_demand_mmn, marginal_prob_disrupt_mmn, SecondMomentProb_mmn, IndexPair_mmn = GenerateMomentDataNoCov(info,
            rawdata_train, info[
                'num_customer'], info['num_facility'])
        loc_chosen_mmn, obj_value_mmn, t_mmn, gap_mmn = SupLazy_SecondCrossNoCov(info, mean_demand_mmn,
                                                                                 marginal_prob_disrupt_mmn,
                                                                                 SecondMomentProb_mmn, IndexPair_mmn)

        # print(loc_chosen_mmn, obj_value_mmn, t_mmn, gap_mmn)
        saa_mmn = SAACost(loc_chosen_mmn, info['fixed_cost'], info['dist'], info['num_customer'], info['max_demand'],rawdata_test)
        str_mmn = ";".join([str(x) for x in loc_chosen_mmn])

        outline2 = ['%s' % 'CM_nocov'] + ['%d' % (t)] + ['%d' % (t+train_length-1)] \
                   +['%d' % (t+train_length)] + ['%d' % (t+train_length+test_length-1)]  \
                   + ['%.2f' % obj_value_mmn] + ['%s'%str_mmn] + ['%.6f' % t_mmn]  \
                  + ['%.2f' % saa_mmn]  +['%.4f'% gap_mmn]
        outputfile.writelines(','.join(outline2) + '\n')

        ##marginal moment method with covariate in Lv 2015
        loc_lvc, obj_lvc, t_lvc, gap_lvc = SolveRobustCov(info, mean_demand_mmc, marginal_prob_mmc, marginal_prob_disrupt_mmc)
        saa_lvc = SAACost(loc_lvc, info['fixed_cost'], info['dist'], info['num_customer'], info['max_demand'], rawdata_test)
        str_lvc =  ";".join([str(x) for x in loc_lvc])

        outline3 = ['%s' % 'MM_cov'] + ['%d' % (t)] + ['%d' % (t + train_length - 1)] \
                   + ['%d' % (t + train_length)] + ['%d' % (t + train_length + test_length - 1)] \
                   + ['%.2f' % obj_lvc] + ['%s' % str_lvc] + ['%.6f' % t_lvc] \
                   + ['%.2f' % saa_lvc] + ['%.4f' % gap_lvc]
        outputfile.writelines(','.join(outline3) + '\n')

        ## Marginal moment robust with no covariate in lv
        loc_lvn, obj_lvn, t_lvn, gap_lvn = SolveRobust(info, mean_demand_mmn, marginal_prob_disrupt_mmn)
        saa_lvn = SAACost(loc_lvn, info['fixed_cost'], info['dist'], info['num_customer'], info['max_demand'], rawdata_test)
        str_lvn =  ";".join([str(x) for x in loc_lvn])

        outline4 = ['%s' % 'MM_nocov'] + ['%d' % (t)] + ['%d' % (t + train_length - 1)] \
                   + ['%d' % (t + train_length)] + ['%d' % (t + train_length + test_length - 1)] \
                   + ['%.2f' % obj_lvn] + ['%s' % str_lvn] + ['%.6f' % t_lvn] \
                   + ['%.2f' % saa_lvn] + ['%.4f' % gap_lvn]
        outputfile.writelines(','.join(outline4) + '\n')


    outputfile.close()
    print("Storm test of Moment-based methods finished!")