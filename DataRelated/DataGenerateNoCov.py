"""
Code for "Hao Shen, Mengying Xue, and Zuojun Max Shen (2023),
Data-driven Reliable Faciltiy Location Problem, Management Science, Forthcoming",

Last Revised on Jan 12, 2024
@author: Mengying Xue
"""
## generate synthetic data for the tests in Section 5


import pandas as pd
from sklearn import datasets
from scipy.stats import multivariate_normal
from scipy.special import erfinv
import math
from Utility.Constants import demand_coeff, NumSimulate, NumTestData,Num_real
import numpy as np
import pickle
from DataRelated.DataGenerate import read_from_txt, MapDistance

def DataGeneration_train_test_normal_truncate(num_J, max_data_length, demand_val, max_demand,  cov_matrix, mu_coeff, cov_threshold):
    num_I = num_J
    mean = [0 for i in range(num_I + num_J)]
    data = multivariate_normal.rvs(mean, cov_matrix, size=max_data_length).reshape(max_data_length, num_I+num_J)

    for t in range(max_data_length):
        for i in range(num_I):
            data[t][i] = min(max(demand_val[i]*(mu_coeff + data[t][i]), 0), max_demand[i])

        for i in range(num_I, num_I + num_J):
            if data[t][i] > math.sqrt(2) * erfinv(2 * cov_threshold - 1):
                ## operate
                data[t][i] = 1
            else:
                ## disrupt
                data[t][i] = 0
    b = np.zeros(max_data_length).reshape(max_data_length, 1)
    data = np.append(data, b, axis=1)
    df = pd.DataFrame(data, columns = ['d_%d' % i for i in range(num_I)] + ['disrupt_%d'%j for j in range(num_J)] + ['cov'])

    return df


def DataGeneration_train_test_lognormal_notruncate(num_J, max_data_length, demand_val, cov_matrix, mu_coeff, cov_threshold):
    num_I = num_J
    mean = [0 for i in range(num_I + num_J)]
    data = multivariate_normal.rvs(mean, cov_matrix, size=max_data_length).reshape(max_data_length, num_I + num_J)

    for t in range(max_data_length):
        for i in range(num_I):
            data[t][i] = demand_val[i] * mu_coeff * np.exp(data[t][i])

        for i in range(num_I, num_I + num_J):
            if data[t][i] > math.sqrt(2) * erfinv(2 * cov_threshold - 1):
                ## operate
                data[t][i] = 1
            else:
                ## disrupt
                data[t][i] = 0
    b = np.zeros(max_data_length).reshape(max_data_length, 1)
    data = np.append(data, b, axis=1)
    df = pd.DataFrame(data,
                      columns=['d_%d' % i for i in range(num_I)] + ['disrupt_%d' % j for j in range(num_J)] + ['cov'])
    return df

## generate the  distributions for states and demands in the no covariate case
def GenerateNoCovDistribution(num_node, mu_coeff):
    file = 'Data/UCFLData%d.txt' % num_node
    mu, f, o, num_I, num_J, coor_I, coor_J = read_from_txt(file)


    distribution = {}
    distribution.update({'cov_matrix': datasets.make_spd_matrix(num_J + num_I)})
    distribution.update({'mu_coeff': mu_coeff})
    distribution.update({'threshold': 0.1})
    print(distribution)
    with open('Data/RawDataDistribution/Node_%d-mu_%f.pkl' % (num_node, mu_coeff), 'wb') as tf:
        pickle.dump(distribution, tf)
    return distribution

## generate the  test data in the no covariate case
def GenerateTestDataNoCovFromDistributionFile(distribution, num_node, test_data_length, truncate):
    cov_matrix = distribution['cov_matrix']
    mu_coeff = distribution['mu_coeff']
    cov_threshold = distribution['threshold']
    file = 'Data/UCFLData%d.txt' % num_node
    mu, f, o, num_I, num_J, coor_I, coor_J = read_from_txt(file)

    max_demand = [mu[i] * demand_coeff for i in range(num_I)]


    test_file_name_lst = []
    if truncate:
        rawdata_test = DataGeneration_train_test_normal_truncate(num_J, test_data_length, mu, max_demand,  cov_matrix, mu_coeff, cov_threshold)
    else:
        rawdata_test = DataGeneration_train_test_lognormal_notruncate(num_J, test_data_length, mu, cov_matrix, mu_coeff,cov_threshold)
    test_file_name = 'Data/RawDataNoCov/Test_Node_%d-mu_%f-truncate_%d-Length_%d.csv' % (num_node, mu_coeff, truncate, test_data_length)
    rawdata_test.to_csv(test_file_name, index=False)
    test_file_name_lst.append(test_file_name)


## generate the  training data in the no covariate case
def GenerateRawTrainDataNoCovfromDistribution(distribution, num_node, max_data_length, truncate):
    file = 'Data/UCFLData%d.txt' % num_node
    mu, f, o, num_I, num_J, coor_I, coor_J = read_from_txt(file)
    max_demand = [mu[i] * demand_coeff for i in range(num_I)]
    train_file_name_lst = []
    cov_matrix = distribution['cov_matrix']
    mu_coeff = distribution['mu_coeff']
    cov_threshold = distribution['threshold']

    for k in range(NumSimulate):
        if truncate:
            rawdata_train = DataGeneration_train_test_normal_truncate(num_J, max_data_length, mu, max_demand,
                                                                     cov_matrix, mu_coeff, cov_threshold)
        else:
            rawdata_train = DataGeneration_train_test_lognormal_notruncate(num_J, max_data_length, mu, cov_matrix, mu_coeff,
                                                                          cov_threshold)
        train_file_name = 'Data/RawDataNoCov/Train_Node_%d-mu_%f-truncate_%d-Simulate_%d.csv' % (num_node, mu_coeff, truncate, k)
        rawdata_train.to_csv(train_file_name, index=False)
        train_file_name_lst.append(train_file_name)

    if truncate:
        with open('Data/RawDataNoCovFileName/Train_Node_%d-mu_%f-truncate_%d.txt' % (num_node, mu_coeff, truncate), 'w') \
                as output_file:
            for name in train_file_name_lst:
                outline = ['%s' % name]
                output_file.write(','.join(outline) + '\n')
    else:
        with open('Data/RawDataNoCovFileName/Train_Node_%d-mu_%f-truncate_%d.txt' % (num_node, 1, truncate), 'w') \
                as output_file:
            for name in train_file_name_lst:
                outline = ['%s' % name]
                output_file.write(','.join(outline) + '\n')

## generate the large data set for approximating the true optimum in the no covariate case
def GenerateSAADataNoCovfromDistribution(distribution, num_node, truncate):
    file = 'Data/UCFLData%d.txt' % num_node
    mu, f, o, num_I, num_J, coor_I, coor_J = read_from_txt(file)
    max_demand = [mu[i] * demand_coeff for i in range(num_I)]
    real_file_name_lst = []

    cov_matrix = distribution['cov_matrix']
    mu_coeff = distribution['mu_coeff']
    cov_threshold = distribution['threshold']

    if truncate:
        rawdata_real = DataGeneration_train_test_normal_truncate(num_J, Num_real, mu, max_demand, cov_matrix,
                                                                 mu_coeff, cov_threshold)
    else:
        rawdata_real = DataGeneration_train_test_lognormal_notruncate(num_J, Num_real, mu, cov_matrix, mu_coeff,
                                                                      cov_threshold)
    real_file_name = 'Data/RawDataNoCov/Real_Node_%d-mu_%f-truncate_%d.csv' % (num_node, mu_coeff, truncate)
    rawdata_real.to_csv(real_file_name, index = False)
    real_file_name_lst.append(real_file_name)

# generate all the data in the no covariate case
def GenerateRawDataNoCovFileList(num_node, max_data_length, mu_coeff, truncate):
    distribution = GenerateNoCovDistribution(num_node, mu_coeff)
    GenerateRawTrainDataNoCovfromDistribution(distribution, num_node, max_data_length, truncate)
    GenerateTestDataNoCovFromDistributionFile(distribution, num_node, max_data_length, truncate)
    GenerateSAADataNoCovfromDistribution(distribution, num_node, truncate)
# read the distribution information in the no covariate case
def ReadNoCovDistribution(num_node, mu_coeff):
    file = 'Data/RawDataDistribution/Node_%d-mu_%f.pkl' % (num_node, mu_coeff)
    input_file = open(file, 'rb')
    data = pickle.load(input_file)

    cov_matrix = data['cov_matrix']
    print(cov_matrix)
    mu_coeff = data['mu_coeff']
    print(mu_coeff)
    cov_threshold = data['threshold']
    print(cov_threshold)
    return data


# read training data the folder /Data/RawDataNoCov in the no covariate case
def ReadRawTrainDataNoCovFromFile(num_node, mu_coeff, truncate, num_data):
    train_file_name = open('Data/RawDataNoCovFileName/Train_Node_%d-mu_%f-truncate_%d.txt'% (num_node, mu_coeff, truncate),'r')

    train_name_lst = []
    for line in train_file_name.readlines():
        line = line.rstrip("\n")
        train_name_lst.append(line)
    # print(train_name_lst)

    train_data_lst = []

    for k in range(len(train_name_lst)):
        train_data_lst.append(pd.read_csv(train_name_lst[k])[:num_data])


    file = 'Data/UCFLData%d.txt' % num_node
    mu, f, o, num_I, num_J, coor_I, coor_J = read_from_txt(file)
    f = f + [0]
    d0 = MapDistance(coor_I, coor_J)
    dist = [d0[i] + [o[i]] for i in range(num_I)]
    max_demand = [mu[i] * demand_coeff for i in range(num_I)]
    info = {'dist': dist, 'fixed_cost': f, 'mu': mu, 'num_customer': num_I, 'num_facility': num_J,
                  'max_demand': max_demand,
                  'num_cov': 1}
    return train_data_lst, info

# read test data the folder /Data/RawDataNoCov in the no covariate case
def ReadRawTestDataNoCovFromFile(num_node, mu_coeff, truncate):
    test_data_name = 'Data/RawDataNoCov/Test_Node_%d-mu_%f-truncate_%d-Length_%d.csv'%(num_node,mu_coeff, truncate, NumTestData)
    test_data = pd.read_csv(test_data_name, low_memory=False)
        # print(test_data_lst[-1].loc[0:10])

    file = 'Data/UCFLData%d.txt' % num_node
    mu, f, o, num_I, num_J, coor_I, coor_J = read_from_txt(file)
    f = f + [0]
    d0 = MapDistance(coor_I, coor_J)
    dist = [d0[i] + [o[i]] for i in range(num_I)]
    max_demand = [mu[i] * demand_coeff for i in range(num_I)]
    info = {'dist': dist, 'fixed_cost': f, 'mu': mu, 'num_customer': num_I, 'num_facility': num_J,
                  'max_demand': max_demand,
                  'num_cov': 1}
    return test_data

# read large data set for approximating the true optimum from the folder /Data/RawDataNoCov in the no covariate case
def ReadRawDataSAANoCov(num_node, mu_coeff, truncate):
    train_file_name = 'Data/RawDataNoCov/Real_Node_%d-mu_%f-truncate_%d.csv'%(num_node, mu_coeff, truncate)

    # train_data = pd.read_csv(train_file_name).loc[0:100000]
    train_data = pd.read_csv(train_file_name)

    file = 'Data/UCFLData%d.txt' % num_node
    mu, f, o, num_I, num_J, coor_I, coor_J = read_from_txt(file)
    f = f + [0]
    d0 = MapDistance(coor_I, coor_J)
    dist = [d0[i] + [o[i]] for i in range(num_I)]
    max_demand = [mu[i] * demand_coeff for i in range(num_I)]
    info = {'dist': dist, 'fixed_cost': f, 'mu': mu, 'num_customer': num_I, 'num_facility': num_J,
                  'max_demand': max_demand,
                  'num_cov': 1}
    return train_data, info
