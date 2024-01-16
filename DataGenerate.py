"""
Code for "Hao Shen, Mengying Xue, and Zuojun Max Shen (2023),
Data-driven Reliable Faciltiy Location Problem, Management Science, Forthcoming",

Last Revised on Jan 12, 2024
@author: Mengying Xue
"""

import pandas as pd
from sklearn import datasets
from scipy.stats import multivariate_normal
from scipy.special import erfinv
import math
from Utility.dist import Distance_NewOrleans, lat_dis
from Utility.Constants import demand_coeff, NumSimulate, NumTestData,Num_real
import numpy as np
import pickle
from Utility.Utils import positive_num



# read the information from the data set in Snyder and Daskin (2005)
def read_from_txt(file):
    mu = []
    o = []
    f = []
    coor_I = []
    coor_J = []
    with open(file, 'r') as file_to_read:
        idx = 0
        for lines in file_to_read:
            line = lines.split("\t")
            if idx == 0:
                num_I = num_J = lines[0]
                idx += 1
            else:
                mu.append(float(line[1]))
                o.append(float(line[2]))
                f.append(float(line[3]))
                coor_I.append((float(line[4]), float(line[5].split('\n')[0])))
                coor_J.append((float(line[4]), float(line[5].split('\n')[0])))
                idx += 1
    num_I = len(coor_I)
    num_J = len(coor_J)
    return mu,  f, o, num_I, num_J,  coor_I, coor_J

# Calculate the distance between two locations
def MapDistance(coor_I, coor_J):
    num_I = len(coor_I)
    num_J = len(coor_J)
    d0 = [[lat_dis(coor_I[i], coor_J[j]) for j in range(num_J)] for i in range(num_I)]
    return d0

# Calculate the distance of a given location to the center of storm in NewOrleans
def DistNewOrleans(coor_J):
    dist_NO = [Distance_NewOrleans(coor_J[j]) for j in range(len(coor_J))]
    return dist_NO

## Generate synthetic data with covariates (for test in Section 6)
def DataGeneration_train_test_Covariate(num_J, max_data_length, demand_val, max_demand, cov_marginal_prob, cov_matrix, mu_coeff, cov_threshold, cov_set):
    num_I = num_J

    mean = [0 for i in range(num_I + num_J)]
    #generate covariate
    cov_data = np.random.choice(cov_set, size = max_data_length, p=cov_marginal_prob)
    # the generated covariate sets
    cov_set_lean = np.unique(cov_data)
    # the number of each coviarates
    cov_num = [np.sum(cov_data == cov) for cov in cov_set]

    # dict stores the index of each covaraite
    cov_position = {}
    for cov in cov_set:
        cov_position.update({cov: np.where(cov_data == cov)[0]})

    raw_data = {}

    data = np.zeros((max_data_length,num_I+num_J+1))

    for cov in cov_set_lean:
        n_size = cov_num[np.where(cov_set == cov)[0][0]]
        raw_data.update({cov: multivariate_normal.rvs(mean, cov_matrix[cov], size=n_size).reshape(n_size, num_I+num_J)})

    for t in range(max_data_length):
        cov = cov_data[t]
        data[t][-1] = cov
        idx = np.where(cov_position[cov] == t)[0][0]

        for i in range(num_I):
            data[t][i] = min(max(demand_val[i]*(mu_coeff[cov] + raw_data[cov][idx][i]), 0), max_demand[i])
        for i in range(num_I, num_I + num_J):
            if raw_data[cov][idx][i] > math.sqrt(2) * erfinv(2 * cov_threshold[cov] - 1):
                ## operate
                data[t][i] = 1
            else:
                ## disrupt
                data[t][i] = 0

    df = pd.DataFrame(data, columns = ['d_%d' % i for i in range(num_I)] + ['disrupt_%d'%j for j in range(num_J)] + ['cov'])
    return df


## generate correlated covariate data
def GenerateDistribution(num_node,num_cov):
    # the location data is obtained from Synder and Daskin's paper
    file = 'Data/UCFLData%d.txt' % num_node
    mu, f, o, num_I, num_J, coor_I, coor_J = read_from_txt(file)

    cov_set = np.array(range(1, num_cov+1))
    ## randomly generate marginal probabilities
    cov_marginal_prob = [i+1 for i in range(num_cov)]
    cov_marginal_prob = cov_marginal_prob/np.sum(cov_marginal_prob)


    cov_matrix = {}
    mu_coeff = {}
    cov_threshold = {}
    for cov in cov_set:
        cov_matrix.update({cov: datasets.make_spd_matrix(num_J + num_I)})
        mu_coeff.update({cov: 0.6/cov + positive_num(cov-1, int(num_cov/2))*1})
        cov_threshold.update({cov: min(0.1 + 0.2 *(cov-1), 0.8)})
    distribution = {}
    distribution.update({'cov_marginal_prob':cov_marginal_prob})
    distribution.update({'cov_matrix': cov_matrix})
    distribution.update({'mu_coeff': mu_coeff})
    distribution.update({'cov_threshold': cov_threshold})

    with open('Data/RawDataDistribution/Node_%d-Cov_%d.pkl'%(num_node, num_cov),'wb') as tf:
        pickle.dump(distribution, tf)
    return distribution


def GenerateTestDataFromDistributionFile(distribution, num_node, num_cov, test_data_length):
    cov_marginal_prob = distribution['cov_marginal_prob']
    cov_matrix = distribution['cov_matrix']
    mu_coeff = distribution['mu_coeff']
    cov_threshold = distribution['cov_threshold']
    file = 'Data/UCFLData%d.txt' % num_node
    mu, f, o, num_I, num_J, coor_I, coor_J = read_from_txt(file)
    cov_set = np.array(range(1, num_cov + 1))
    max_demand = [mu[i] * demand_coeff for i in range(num_I)]

    test_file_name_lst = []
    rawdata_test= DataGeneration_train_test_Covariate(num_J, test_data_length, mu, max_demand, cov_marginal_prob,
                                                        cov_matrix, mu_coeff, cov_threshold, cov_set)
    test_file_name = 'Data/RawData/Test_Node_%d-Cov_%d-Length_%d.csv' % (num_node, num_cov, test_data_length)
    rawdata_test.to_csv(test_file_name, index = False)
    test_file_name_lst.append(test_file_name)

    with open('Data/RawDataFileName/Test_Node_%d-Cov_%d-Length_%d.txt'% (num_node, num_cov, test_data_length), 'w') \
            as output_file:
        for name in test_file_name_lst:
            outline = ['%s'%name]
            output_file.write(','.join(outline) + '\n')


def GenerateRawTrainDatafromDistribution(distribution, num_node, num_cov, max_data_length):
    file = 'Data/UCFLData%d.txt' % num_node
    mu, f, o, num_I, num_J, coor_I, coor_J = read_from_txt(file)
    cov_set = np.array(range(1, num_cov + 1))
    max_demand = [mu[i] * demand_coeff for i in range(num_I)]
    train_file_name_lst = []
    test_file_name_lst = []
    real_file_name_lst = []

    cov_marginal_prob = distribution['cov_marginal_prob']
    cov_matrix = distribution['cov_matrix']
    mu_coeff = distribution['mu_coeff']
    cov_threshold = distribution['cov_threshold']

    for k in range(NumSimulate):
        # rawdata_train, rawdata_test = DataGeneration_train_test(num_J, num_data, mu, max_demand, num_cov)
        rawdata_train = DataGeneration_train_test_Covariate(num_J, max_data_length, mu, max_demand, cov_marginal_prob,
                                                            cov_matrix, mu_coeff, cov_threshold, cov_set)
        train_file_name = 'Data/RawData/Train_Node_%d-Cov_%d-Simulate_%d.csv'% ( num_node, num_cov, k)
        rawdata_train.to_csv(train_file_name, index = False)
        train_file_name_lst.append(train_file_name)



    with open('Data/RawDataFileName/Train_Node_%d-Cov_%d.txt'% (num_node, num_cov), 'w') \
            as output_file:
        for name in train_file_name_lst:
            outline = ['%s'%name]
            output_file.write(','.join(outline) + '\n')

## generate synthetic data for sample average approximation method
def GenerateSAADatafromDistribution(distribution, num_node, num_cov):
    file = 'Data/UCFLData%d.txt' % num_node
    mu, f, o, num_I, num_J, coor_I, coor_J = read_from_txt(file)
    cov_set = np.array(range(1, num_cov + 1))
    max_demand = [mu[i] * demand_coeff for i in range(num_I)]
    real_file_name_lst = []

    cov_marginal_prob = distribution['cov_marginal_prob']
    cov_matrix = distribution['cov_matrix']
    mu_coeff = distribution['mu_coeff']
    cov_threshold = distribution['cov_threshold']


    rawdata_real= DataGeneration_train_test_Covariate(num_J, Num_real, mu, max_demand, cov_marginal_prob,
                                                        cov_matrix, mu_coeff, cov_threshold, cov_set)
    real_file_name = 'Data/RawData/Real_Node_%d-Cov_%d.csv' % (num_node, num_cov)
    rawdata_real.to_csv(real_file_name, index = False)
    real_file_name_lst.append(real_file_name)

    with open('Data/RawDataFileName/Real_Node_%d-Cov_%d.txt'% (num_node, num_cov), 'w') \
            as output_file:
        for name in real_file_name_lst:
            outline = ['%s' % name]
            output_file.write(','.join(outline) + '\n')

def GenerateRawDataFileList(num_node, num_cov, max_data_length):
    distribution = GenerateDistribution(num_node, num_cov)
    GenerateRawTrainDatafromDistribution(distribution, num_node, num_cov, max_data_length)
    GenerateTestDataFromDistributionFile(distribution, num_node, num_cov, NumTestData)
## read distribution from file


def ReadDistribution(num_node, num_cov):
    file = 'Data/RawDataDistribution/Node_%d-Cov_%d.pkl' % (num_node, num_cov)
    input_file = open(file, 'rb')
    data = pickle.load(input_file)

    cov_marginal_prob = data['cov_marginal_prob']
    # print(cov_marginal_prob)
    cov_matrix = data['cov_matrix']
    # print(cov_matrix)
    mu_coeff = data['mu_coeff']
    # print(mu_coeff)
    cov_threshold = data['cov_threshold']
    # print(cov_threshold)
    return data

def GenerateSAAData(num_node, num_cov):
    distribution = ReadDistribution(num_node, num_cov)
    GenerateSAADatafromDistribution(distribution, num_node, num_cov)

def ReadRawTrainDataFromFile(num_node, num_data, num_cov):
    train_file_name = open('Data/RawDataFileName/Train_Node_%d-Cov_%d.txt'% (num_node, num_cov),'r')

    train_name_lst = []
    for line in train_file_name.readlines():
        line = line.rstrip("\n")
        train_name_lst.append(line)
    # print(train_name_lst)


    train_data_lst = []

    for k in range(len(train_name_lst)):
        train_data_lst.append(pd.read_csv(train_name_lst[k])[:num_data])
        # print(train_data_lst[-1])

        # print(test_data_lst[-1].loc[0:10])

    file = 'Data/UCFLData%d.txt' % num_node
    mu, f, o, num_I, num_J, coor_I, coor_J = read_from_txt(file)
    f = f + [0]
    d0 = MapDistance(coor_I, coor_J)
    dist = [d0[i] + [o[i]] for i in range(num_I)]
    max_demand = [mu[i] * demand_coeff for i in range(num_I)]
    info = {'dist': dist, 'fixed_cost': f, 'mu': mu, 'num_customer': num_I, 'num_facility': num_J,
                  'max_demand': max_demand,
                  'num_cov': num_cov}
    return train_data_lst, info

def ReadRawTestDataFromFile(num_node, num_cov):

    test_data_name = 'Data/RawData/Test_Node_%d-Cov_%d-Length_%d.csv'%(num_node, num_cov, NumTestData)
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
                  'num_cov': num_cov}
    return test_data


def ReadRawDataSAA(num_node, num_cov):
    train_file_name = 'Data/RawData/Real_Node_%d-Cov_%d.csv'%(num_node, num_cov)

    train_data = pd.read_csv(train_file_name)
    file = 'Data/UCFLData%d.txt' % num_node
    mu, f, o, num_I, num_J, coor_I, coor_J = read_from_txt(file)
    f = f + [0]
    d0 = MapDistance(coor_I, coor_J)
    dist = [d0[i] + [o[i]] for i in range(num_I)]
    max_demand = [mu[i] * demand_coeff for i in range(num_I)]
    info = {'dist': dist, 'fixed_cost': f, 'mu': mu, 'num_customer': num_I, 'num_facility': num_J,
                  'max_demand': max_demand,
                  'num_cov': num_cov}
    return train_data, info
