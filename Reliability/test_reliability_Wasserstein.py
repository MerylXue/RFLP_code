from  Utility.Constants import NumSimulate, START_YEAR, END_YEAR
from SampleAverage.SAAFunc import SAACost
from datetime import date
import time
import numpy as np
from Reliability.Reliability_Wass import GetOptEpsWasser
from Wasserstein.Wasserstein import WassersteinOpt

def run_Wass_eps(rawdata_train_lst, rawdata_test, info, num_node, num_data, num_cov, beta_, eps, mu_coeff, truncate):
    sim_data_driven_cost = np.zeros(NumSimulate)
    sim_out_of_sample_saa = np.zeros(NumSimulate)
    reliability = 0.0
    time_sol = 0.0
    avg_gap = 0.0
    outputfile_wasser = open('result/Reliability/Wasserstein_%d_%d_%f_Eps%f_Mu_%f_Truncate_%d.csv' % (
        num_data, num_node, beta_,eps, mu_coeff, truncate), 'w')

    outline2 = ['index', 'in_sample_cost', 'out_of_sample_cost',  'Str_loc',  'sol_time', 'gap']

    outputfile_wasser.writelines(','.join(outline2) + '\n')
    for l in range(NumSimulate):
        # read the data
        rawdata_train = rawdata_train_lst[l]
        # print(rawdata_train)
        # print(rawdata_test)
        # update the number of cov in info
        info['num_cov'] = len(list(set(rawdata_train['cov'])))
        loc_wasser, obj_wasser, t_wasser, gap_wasser = WassersteinOpt(rawdata_train, info, eps)
        str_loc_wasser = ";".join([str(x) for x in loc_wasser])
        time_sol += t_wasser
        avg_gap  += gap_wasser
        saa_wasser = SAACost(loc_wasser, info['fixed_cost'], info['dist'], num_node, info['max_demand'],  rawdata_test)

        if saa_wasser <= obj_wasser:
            reliability  += 1
        outline_wasser = ["%d" % l] + ["%.2f" % obj_wasser] + ["%.2f" % saa_wasser]  + ['%s' % str_loc_wasser] \
                          + ['%.2f' % t_wasser] + ['%f' % gap_wasser]
        outputfile_wasser.writelines(','.join(outline_wasser) + '\n')
        sim_data_driven_cost[l] = obj_wasser
        sim_out_of_sample_saa[l] = saa_wasser

    # sim_data_driven_cost = sim_data_driven_cost  / NumSimulate
    mean_in_sample_cost = np.average(sim_data_driven_cost)
    q1_in_sample_cost = np.quantile(sim_data_driven_cost, q=0.1)
    q2_in_sample_cost = np.quantile(sim_data_driven_cost, q=0.2)
    q8_in_sample_cost = np.quantile(sim_data_driven_cost, q=0.8)
    q9_in_sample_cost = np.quantile(sim_data_driven_cost, q=0.9)

    mean_out_of_sample_saa =np.average(sim_out_of_sample_saa)
    q1_out_of_sample_saa = np.quantile(sim_out_of_sample_saa, q = 0.1)
    q2_out_of_sample_saa = np.quantile(sim_out_of_sample_saa, q = 0.2)
    q8_out_of_sample_saa = np.quantile(sim_out_of_sample_saa, q = 0.8)
    q9_out_of_sample_saa = np.quantile(sim_out_of_sample_saa, q = 0.9)

    reliability = reliability / NumSimulate
    time_sol = time_sol/ NumSimulate
    avg_gap = avg_gap / NumSimulate

    ## output reliability result
    outputfile = open('result/Reliability_Wasserstein_Node%d_Data%d_Cov%d_Beta%.4f_Eps%f_Mu_%f_Truncate_%d.csv'
                      % (num_node, num_data, num_cov, beta_,eps, mu_coeff, truncate), 'w')
    out_line_col0 = ['num_data', 'Method', 'in_sample_cost_avg', 'in_sample_cost_q1',
                     'in_sample_cost_q2', 'in_sample_cost_q8', 'in_sample_cost_q9',
                     'out_of_sample_cost_avg','out_of_sample_cost_q1',
                     'out_of_sample_cost_q2', 'out_of_sample_cost_q8', 'out_of_sample_cost_q9',
                     'out_of_sample_reliability',   'time_sol', 'Gap']
    outputfile.writelines(','.join(out_line_col0) + '\n')

    outline = ['%d' % num_data]
    outline += ['Wasserstein'] \
               + ['%.2f' % mean_in_sample_cost] + ['%.2f' % q1_in_sample_cost] + ['%.2f' % q2_in_sample_cost] \
               + ['%.2f' % q8_in_sample_cost] + ['%.2f' % q9_in_sample_cost] \
               + ['%.2f' % mean_out_of_sample_saa] + ['%.2f' % q1_out_of_sample_saa] + ['%.2f' % q2_out_of_sample_saa] \
               + ['%.2f' % q8_out_of_sample_saa] + ['%.2f' % q9_out_of_sample_saa] \
               + ['%.4f' % reliability] + ['%.4f' % time_sol] + ['%.4f' % avg_gap]
    outputfile.writelines(','.join(outline) + '\n')

    outputfile.close()
    outputfile_wasser.close()
    print("Reliability test of wasserstein finished!")


def test_Reliability_Wasserstein(rawdata_train_lst, rawdata_test, info, num_node, num_data, num_cov, beta_, mu_coeff, truncate):

    Opt_EpsSet = 0
    # Opt_EpsSamplesSet = [[0 for k in range(num_cov)] for i in range(n_method)]
    sim_data_driven_cost = np.zeros(NumSimulate)
    sim_out_of_sample_saa =np.zeros(NumSimulate)
    in_sample_rel = 0
    reliability = 0
    count_search = 0
    time_search = 0
    time_sol = 0
    avg_gap = 0


    outputfile_wasser = open('result/Reliability/Wasserstein_data%d_node%d_cov%d_beta%f_Mu_%f_Truncate_%d.csv' % (
             num_data, num_node, num_cov, beta_, mu_coeff, truncate), 'w')



    outline2 = ['index', 'in_sample_cost', 'out_of_sample_cost', 'in_sample_rel', 'Str_loc', 'Eps',
                'Eps_Search_Time', 'Eps_all_time', 'num_search', 'sol_time', 'gap']

    outputfile_wasser.writelines(','.join(outline2) + '\n')
    for l in range(NumSimulate):
    # for l in [50]:
    #     print(l)
        # read the data
        rawdata_train = rawdata_train_lst[l]
        # rawdata_test = rawdata_test_lst[l]
        # print(rawdata_train, rawdata_test)
        # print(rawdata_train)
        # print(rawdata_test)
        # update the number of cov in info
        info['num_cov'] = len(list(set(rawdata_train['cov'])))

        ## wasserstein
        t3  = time.time()
        opt_eps_wasser, avg_out_of_sample_lst_wasser, t_eps_wasser, count_wasser = GetOptEpsWasser(info, beta_,
                                                                                               rawdata_train)
        t4 = time.time()
        count_search += count_wasser
        time_search += t_eps_wasser


        loc_wasser, obj_wasser, t_wasser, gap_wasser = WassersteinOpt(rawdata_train,info,  opt_eps_wasser)
        print(loc_wasser, obj_wasser)
        str_loc = ";".join([str(x) for x in loc_wasser])
        time_sol  += t_wasser
        avg_gap  += gap_wasser
        saa_wasser = SAACost(loc_wasser, info['fixed_cost'], info['dist'], num_node, info['max_demand'], rawdata_test)
        # print(loc_wasser)

        if saa_wasser <= obj_wasser:
            # print(saa_wasser, obj_wasser)
            reliability  += 1

        rel_wasser = 0
        for item in avg_out_of_sample_lst_wasser:
            if item < obj_wasser:
                rel_wasser += 1

        if len(avg_out_of_sample_lst_wasser) > 0:
            opt_rel_wasser_in = rel_wasser / len(avg_out_of_sample_lst_wasser)
        else:
            opt_rel_wasser_in = 0
        # record the details of each simulation in separete file
        outline_wasser = ["%d" % l] + ["%.2f" % obj_wasser] + ["%.2f" % saa_wasser] + ["%.4f" % opt_rel_wasser_in] \
                      + ['%s' % str_loc] + ["%f" % opt_eps_wasser] + ['%.2f' % t_eps_wasser] \
                         + ['%.2f' % (t4 - t3)] + ['%d' % count_wasser] + ['%.2f' % t_wasser] + [ '%f' % gap_wasser]
        outputfile_wasser.writelines(','.join(outline_wasser) + '\n')

        # record the parameter and results to the upper level group
        Opt_EpsSet  += opt_eps_wasser
        sim_data_driven_cost[l] = obj_wasser
        sim_out_of_sample_saa[l] = saa_wasser
        in_sample_rel += opt_rel_wasser_in


    Opt_EpsSet = Opt_EpsSet/NumSimulate
    # sim_data_driven_cost = sim_data_driven_cost / NumSimulate
    # sim_out_of_sample_saa[i] = sim_out_of_sample_saa[i] / NumSimulate
    mean_in_sample_cost = np.average(sim_data_driven_cost)
    q1_in_sample_cost = np.quantile(sim_data_driven_cost, q=0.1)
    q2_in_sample_cost = np.quantile(sim_data_driven_cost, q=0.2)
    q8_in_sample_cost = np.quantile(sim_data_driven_cost, q=0.8)
    q9_in_sample_cost = np.quantile(sim_data_driven_cost, q=0.9)

    mean_out_of_sample_saa = np.average(sim_out_of_sample_saa)
    q1_out_of_sample_saa = np.quantile(sim_out_of_sample_saa, q = 0.1)
    q2_out_of_sample_saa = np.quantile(sim_out_of_sample_saa, q = 0.2)
    q8_out_of_sample_saa = np.quantile(sim_out_of_sample_saa, q = 0.8)
    q9_out_of_sample_saa = np.quantile(sim_out_of_sample_saa, q = 0.9)
    in_sample_rel  = in_sample_rel / NumSimulate
    reliability = reliability / NumSimulate
    count_search = count_search/ NumSimulate
    time_search  = time_search /NumSimulate
    time_sol  = time_sol /NumSimulate
    avg_gap = avg_gap/NumSimulate

    ## output reliability result
    outputfile = open(
        'result/Reliability_Wasserstein_Node%d_Data%d_Cov%d_Beta%.4f_Mu_%f_Truncate_%d.csv' % (num_node, num_data, num_cov, beta_, mu_coeff, truncate),
        'w')
    out_line_col0 = ['num_data', 'Method', 'count_search', 'time_search',
                     'in_sample_cost_avg', 'in_sample_cost_q1',
                     'in_sample_cost_q2', 'in_sample_cost_q8', 'in_sample_cost_q9',
                     'out_of_sample_cost_avg','out_of_sample_cost_q1','out_of_sample_cost_q2',
                     'out_of_sample_cost_q8','out_of_sample_cost_q9',
                     'in_sample_reliability',
                     'out_of_sample_reliability', 'Avg_eps', 'time_sol', 'Gap']
    outputfile.writelines(','.join(out_line_col0) + '\n')
    Method = 'Wasserstein'
    outline = ['%d' % num_data]
    outline += ['%s' % Method] + ['%d' % count_search] + ["%f" % time_search ] \
               + ['%.2f' % mean_in_sample_cost] + ['%.2f' % q1_in_sample_cost] + ['%.2f' % q2_in_sample_cost] \
               + ['%.2f' % q8_in_sample_cost] + ['%.2f' % q9_in_sample_cost] + ['%.2f' % mean_out_of_sample_saa] \
           + ['%.2f' % q1_out_of_sample_saa] + ['%.2f' % q2_out_of_sample_saa] + ['%.2f' % q8_out_of_sample_saa] \
           + ['%.2f' % q9_out_of_sample_saa]  + ['%.4f' % in_sample_rel ] \
           + ['%.4f' % reliability] + ['%f' % Opt_EpsSet ]+ ["%f"%time_sol ]+['%.4f'%avg_gap ]

    outputfile.writelines(','.join(outline) + '\n')

    outputfile.close()
    outputfile_wasser.close()
    print("Reliability test of wasserstein finished!")
    # return sim_data_driven_cost, sim_out_of_sample_saa, in_sample_rel, reliability, Opt_EpsSet, Opt_EpsSamplesSet, count_search, time_search



def run_Storm_Wasserstein(train_data_lst, test_data_lst, info, num_node, num_cov, beta_,  train_length, test_length):
    outputfile = open(
        'result/Storm/Wasserstein-Node_%d-Cov_%d-Beta_%.2f.csv' % (num_node, num_cov,  beta_),
        'w')

    out_line_col0 = ['Method', 'Train_Start_year', 'Train_End_year', 'Test_Start_year', 'Test_End_year',
                      'obj', 'location', 'search_time', 'count_search', 'sol_time', 'OutofSample', 'Gap',
                     'OptEps']
    outputfile.writelines(','.join(out_line_col0) + '\n')
    # print(len(train_data_lst))


    for t in range(START_YEAR, END_YEAR + 1 - train_length):
        rawdata_train = train_data_lst[t-START_YEAR]
        rawdata_test = test_data_lst[t-START_YEAR]

        info['num_cov'] = len(list(set(rawdata_train['cov'])))
        # # generate the train and test data
        # rawdata_train, rawdata_test, info_train, info_test = RawDataGenerate_train_test(num_node, num_data, num_cov)

        # the kolomogorov with covariate

        # search the optimal set of epsilon for kolmogorov method
        opt_eps_wasser, avg_out_of_sample_lst_wasser, t_eps_wasser, count_wasser = GetOptEpsWasser(info, beta_,
                                                                                                   rawdata_train)

        # generate the decision using the optimal eps
        loc_wasser, obj_wasser, t_wasser, gap_wasser = WassersteinOpt(rawdata_train,info,  opt_eps_wasser)

        str_loc = ";".join([str(x) for x in loc_wasser])

        # calculate the out of sample cost
        saa_wasser = SAACost(loc_wasser, info['fixed_cost'], info['dist'], num_node,info['max_demand'], rawdata_test)

        #
        # outputfile.close()

        outline1 =['%s' % 'Wasserstein'] + ['%d' % (t)] + ['%d' % (t+train_length-1)] \
                   +['%d' % (t+train_length)] + ['%d' % (t+train_length+test_length-1)]  \
                   + ['%.2f' % obj_wasser] + ['%s'%str_loc] + ['%.6f' % t_eps_wasser] + ['%d'%count_wasser] + ['%.6f' % t_wasser]  \
                  + ['%.2f' % saa_wasser]  +['%.4f'% gap_wasser] + ['%f' % opt_eps_wasser]

        outputfile.writelines(','.join(outline1) + '\n')


    outputfile.close()
    print("Storm test of Wasserstein finished!")