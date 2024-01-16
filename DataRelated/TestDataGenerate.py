#
# from Utility.Constants import demand_coeff
# from DataRelated.DataGenerate import read_from_txt, MapDistance
# import pandas as pd
#
#
# ##test
#
#
#
#
#
# def ReadTestDataFromFile(num_node, num_cov, test_data_length):
#     test_file_name = open('Data/RawDataFileName/Test_Node_%d-Cov_%d-Length_%d.txt'% (num_node, num_cov, test_data_length), 'r')
#     test_name_lst = []
#     for line in test_file_name.readlines():
#         line = line.rstrip("\n")
#         test_name_lst.append(line)
#     test_data_lst = []
#     for k in range(len(test_name_lst)):
#
#         test_data_lst.append(pd.read_csv(test_name_lst[k]))
#
#     file = 'Data/UCFLData%d.txt' % num_node
#     mu, f, o, num_I, num_J, coor_I, coor_J = read_from_txt(file)
#     f = f + [0]
#     d0 = MapDistance(coor_I, coor_J)
#     dist = [d0[i] + [o[i]] for i in range(num_I)]
#     max_demand = [mu[i] * demand_coeff for i in range(num_I)]
#     info = {'dist': dist, 'fixed_cost': f, 'mu': mu, 'num_customer': num_I, 'num_facility': num_J,
#             'max_demand': max_demand,
#             'num_cov': num_cov}
#     return test_data_lst, info
#
#
#
# def ReadTrainDataFromFile(num_node, num_cov, max_data_length):
#     file_name = open('Data/RawDataFileName/Train_Node_%d-Cov_%d-Length_%d.txt'% (num_node, num_cov, max_data_length), 'r')
#     name_lst = []
#     for line in file_name.readlines():
#         line = line.rstrip("\n")
#         name_lst.append(line)
#     data_lst = []
#     for k in range(len(name_lst)):
#
#         data_lst.append(pd.read_csv(name_lst[k]))
#
#     file = 'Data/UCFLData%d.txt' % num_node
#     mu, f, o, num_I, num_J, coor_I, coor_J = read_from_txt(file)
#     f = f + [0]
#     d0 = MapDistance(coor_I, coor_J)
#     dist = [d0[i] + [o[i]] for i in range(num_I)]
#     max_demand = [mu[i] * demand_coeff for i in range(num_I)]
#     info = {'dist': dist, 'fixed_cost': f, 'mu': mu, 'num_customer': num_I, 'num_facility': num_J,
#             'max_demand': max_demand,
#             'num_cov': num_cov}
#     return data_lst, info
#
#
# def PerformanceSummary(num_node, num_cov, num_data, beta_, method):
#     file_name = open('result/Reliability/%s_ReOut_%d_%d_%d_%f.csv'% (method, num_node, num_data, num_cov, beta_), 'r')
#     data = pd.read_csv(file_name, low_memory=False)
#
#     outputfile = open(
#         'result/Reliability_%s_ReOut_Node%d_Data%d_Cov%d_Beta%.4f.csv' % (method, num_node, num_data, num_cov, beta_),
#         'w')
#
#     out_line_col0 = ['num_data', 'Method', 'in_sample_cost_avg',
#                      'out_of_sample_cost_avg',
#                      'out_of_sample_reliability' ]
#     outputfile.writelines(','.join(out_line_col0) + '\n')
#
#     out_of_sample_reliability =float(len(data.loc[lambda x: x['out_of_sample_cost'] <= x['in_sample_cost']])/len(data))
#     outline = ['%d' % num_data] + ['%s'% method] + ['%.2f' % data['in_sample_cost'].mean()]\
#               + ['%.2f' % data['out_of_sample_cost'].mean()]+  ['%.4f' % out_of_sample_reliability]
#
#     outputfile.writelines(','.join(outline) + '\n')
#
#     outputfile.close()
#
#
#
#
# def PerformanceSummaryKolByCov(num_node, num_cov, num_data, beta_, method):
#     for k in range(1, num_cov + 1):
#         file_name = open('result/Reliability/%s_bycov_node%d_data%d_cov%d|%d_beta%.4f.csv'%
#                          (method, num_node, num_data, k, num_cov, beta_), 'r')
#         data = pd.read_csv(file_name, low_memory=False)
#
#         outputfile = open(
#             'result/Reliability_%s_bycov_node%d_data%d_cov%d|%d_beta%.4f.csv' % (method, num_node, num_data,k, num_cov, beta_),
#             'w')
#
#         out_line_col0 = ['num_data', 'Method', 'in_sample_cost_avg',
#                          'out_of_sample_cost_avg',
#                          'out_of_sample_reliability' ]
#         outputfile.writelines(','.join(out_line_col0) + '\n')
#
#         out_of_sample_reliability =float(len(data.loc[lambda x: x['out_of_sample_cost'] <= x['in_sample_cost']])/len(data))
#         outline = ['%d' % num_data] + ['%s'% method] + ['%.2f' % data['in_sample_cost'].mean()]\
#                   + ['%.2f' % data['out_of_sample_cost'].mean()]+  ['%.4f' % out_of_sample_reliability]
#
#         outputfile.writelines(','.join(outline) + '\n')
#         outputfile.close()
