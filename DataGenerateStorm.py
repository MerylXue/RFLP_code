"""
Code for "Hao Shen, Mengying Xue, and Zuojun Max Shen (2023),
Data-driven Reliable Faciltiy Location Problem, Management Science, Forthcoming",

Last Revised on Jan 12, 2024
@author: Mengying Xue
"""

#How to design the demand value under real disruption?
# When disruption happens, the demand has prob=(0.3,0.4,0.4)
# when disruption does not happen,  the demand has prob=(0.5,0.3,0.2)

import numpy as np
import pandas as pd
import xlrd
import json
from DataRelated.DataGenerate import MapDistance
from Utility.Constants import demand_coeff, START_YEAR, END_YEAR

def read_excel(file):
    wb = xlrd.open_workbook(filename=file)
    sheet1 = wb.sheet_by_index(0)
    # Get the total number of locatons

    rows = sheet1.row_values(2)  # 获取行内容
    num_J = int(rows[1])
    num_I = num_J

    col_index = sheet1.col_values(0)  # 获取列内容
    col_mu = sheet1.col_values(1)
    col_o = sheet1.col_values(2)
    col_f = sheet1.col_values(4)
    col_lat = sheet1.col_values(5)
    col_lon = sheet1.col_values(6)
    mu = [col_mu[j + 3] for j in range(num_J)]
    f = [col_f[j + 3] for j in range(num_J)]
    coor_I = [(col_lat[j + 3], col_lon[j + 3]) for j in range(num_J)]
    coor_J = coor_I

    o = [col_o[j + 3] for j in range(num_J)]

    return mu, f, o, num_I, num_J, coor_I, coor_J
def DataGenerationStormSce(num_J, js, cov_lst, begin_year, end_year, mu, low_demand, high_demand):
    num_I = num_J
    start_time = (begin_year - START_YEAR) * 12
    end_time = (end_year - START_YEAR) * 12 + 12 - 1
    data = []
    for cov_sublist in cov_lst:
        for t in range(start_time, end_time+1):

            data_temp = [0 for i in range(num_I+num_J+1)]
            data_temp[-1] = cov_lst.index(cov_sublist) + 1
            for j in range(num_J):
                if sum([js[cov][t][j] for cov in cov_sublist]) == 0: ## no disruption
                    # data_temp[j] = np.random.choice([low_demand, 1, high_demand], p= high_prob) * mu[j]
                    data_temp[j] = max(min(np.random.normal(high_demand/2), high_demand),0) * mu[j]
                    data_temp[num_I+j] = 1
                # if disrupted, the demand follows a distribution with low probability to take high value
                else:
                    # data_temp[j] = np.random.choice([low_demand, 1, high_demand], p=low_prob) * mu[j]
                    data_temp[j] = max(min(np.random.normal(low_demand/2), high_demand),0) * mu[j]
                    data_temp[num_I + j] = 0
            data.append(data_temp)
    df = pd.DataFrame(data,
                      columns=['d_%d' % i for i in range(num_I)] + ['disrupt_%d' % j for j in range(num_J)] + ['cov'])

    return df

def DataGenerationStormScePD(num_J, df, cov_lst, begin_year, end_year, mu, low_demand, high_demand):
    num_I = num_J

    df = df[(df['start_year'] >= begin_year) & (df['end_year'] <= end_year)].reset_index(drop = True)
    data = []
    # print(df)
    for cov_sublist in cov_lst:
        for t in range(len(df)):
            data_temp = [0 for i in range(num_I+num_J+1)]
            data_temp[-1] = cov_lst.index(cov_sublist) + 1
            for j in range(num_J):
                if df.iloc[t]['%d'%j]  == 0: ## no disruption
                    # data_temp[j] = np.random.choice([low_demand, 1, high_demand], p= high_prob) * mu[j]
                    data_temp[j] = max(min(np.random.normal(high_demand/2), high_demand),0) * mu[j]
                    data_temp[num_I+j] = 1
                # if disrupted, the demand follows a distribution with low probability to take high value
                else:
                    # data_temp[j] = np.random.choice([low_demand, 1, high_demand], p=low_prob) * mu[j]
                    data_temp[j]  = max(min(np.random.normal(low_demand/2), high_demand),0) * mu[j]
                    data_temp[num_I + j] = 0
            data.append(data_temp)

    #
    df2 = pd.DataFrame(data,
                      columns=['d_%d' % i for i in range(num_I)] + ['disrupt_%d' % j for j in range(num_J)] + ['cov'])

    return df2

def RawDataGenerateStormFileList(num_node, train_length, test_length, low_demand, high_demand):
    disrupt_file = open('Data/Storm/disruption_%d.json' % num_node, 'r')
    disruption_js = json.load(disrupt_file)
    event_lst = list(disruption_js.keys())


    cov_lst = [[s for s in event_lst if 'Wind' in s], [s for s in event_lst if not 'Wind' in s]]
    print(cov_lst)


    file = 'Data/data%dUFLP.xls' % num_node
    mu, f, o, num_I, num_J, coor_I, coor_J = read_excel(file)

    f = f + [0]
    d0 = MapDistance(coor_I, coor_J)
    dist = [d0[i] + [o[i]] for i in range(num_I)]


    train_file_name_lst = []
    test_file_name_lst = []
    for t in range(START_YEAR, END_YEAR + 1 - train_length):
        raw_data_train = DataGenerationStormSce(num_J, disruption_js, cov_lst, t, t + train_length - 1, mu, low_demand, high_demand)

        raw_data_test = DataGenerationStormSce(num_J, disruption_js, cov_lst, t + train_length, t + train_length + test_length - 1, mu,  low_demand, high_demand)
        train_file_name = 'Data/RawDataStorm/Train-TrainLength_%d-TestLength_%d-Node_%d-Cov_%d-Year_%d.csv'%\
                          (train_length, test_length, num_node, len(cov_lst), t)
        # print(len(raw_data_train), len(raw_data_test))
        raw_data_train.to_csv(train_file_name, index=False)
        train_file_name_lst.append(train_file_name)

        test_file_name = 'Data/RawDataStorm/Test-TrainLength_%d-TestLength_%d-Node_%d-Cov_%d-Year_%d.csv' %\
                         (train_length, test_length,  num_node, len(cov_lst), t)
        raw_data_test.to_csv(test_file_name, index=False)
        test_file_name_lst.append(test_file_name)

    with open('Data/RawDataStormFileName/Train-TrainLength_%d-TestLength_%d-Node_%d-Cov_%d.txt'%
              (train_length, test_length, num_node, len(cov_lst)), 'w') \
            as output_file:
        for name in train_file_name_lst:
            outline = ['%s'%name]
            output_file.write(','.join(outline) + '\n')
        # output_file.writelines('\n')

    with open('Data/RawDataStormFileName/Test-TrainLength_%d-TestLength_%d-Node_%d-Cov_%d.txt'%
              (train_length, test_length, num_node, len(cov_lst)), 'w') \
            as output_file:
        for name in test_file_name_lst:
            outline = ['%s'%name]
            output_file.write(','.join(outline) + '\n')


def ReadRawDataStormFromFile(num_node, num_cov, train_length, test_length):
    train_file_name = open('Data/RawDataStormFileName/Train-TrainLength_%d-TestLength_%d-Node_%d-Cov_%d.txt'%
                           (train_length, test_length, num_node, num_cov),'r')

    train_name_lst = []
    for line in train_file_name.readlines():
        line = line.rstrip("\n")
        train_name_lst.append(line)
    # print(train_name_lst)
    test_file_name = open('Data/RawDataStormFileName/Test-TrainLength_%d-TestLength_%d-Node_%d-Cov_%d.txt'%
                          (train_length, test_length, num_node, num_cov),'r')
    test_name_lst = []
    for line in test_file_name.readlines():
        line = line.rstrip("\n")
        test_name_lst.append(line)

    train_data_lst = []
    test_data_lst = []
    for k in range(len(train_name_lst)):
        train_data_lst.append(pd.read_csv(train_name_lst[k]))
        test_data_lst.append(pd.read_csv(test_name_lst[k]))


    file = 'Data/data%dUFLP.xls' % num_node
    mu, f, o, num_I, num_J, coor_I, coor_J = read_excel(file)
    f = f + [0]
    d0 = MapDistance(coor_I, coor_J)
    dist = [d0[i] + [o[i]] for i in range(num_I)]
    max_demand = [mu[i] * demand_coeff for i in range(num_I)]
    info = {'dist': dist, 'fixed_cost': f, 'mu': mu, 'num_customer': num_I, 'num_facility': num_J,
                  'max_demand': max_demand,
                  'num_cov': num_cov}
    return train_data_lst, test_data_lst, info

