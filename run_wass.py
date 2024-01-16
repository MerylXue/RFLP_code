"""
Code for "Hao Shen, Mengying Xue, and Zuojun Max Shen (2023),
Data-driven Reliable Faciltiy Location Problem, Management Science, Forthcoming",

Last Revised on Jan 12, 2024
@author: Mengying Xue

"""
## This file is the main function for testing wasserstein based method.
## In the manuscript, we only test Wasserstein based method in no-covariate case



from DataRelated.DataGenerate import  ReadRawTrainDataFromFile, ReadRawTestDataFromFile
from Reliability.test_reliability_Wass import test_Reliability_Wasserstein
def run_PUB():
    num_node = 10
    beta_lst = [0.1]
    num_cov_set = [2]
    NumDataSet = [500]
    ## beta = 0.25, data = 100


    for num_data in NumDataSet:
        for num_cov in num_cov_set:
            for beta_ in beta_lst:
                train_data_lst, info = ReadRawTrainDataFromFile(num_node, num_data, num_cov)
                test_data = ReadRawTestDataFromFile(num_node, num_cov)
                test_Reliability_Wasserstein(train_data_lst, test_data, info, num_node, num_data, num_cov, beta_)
run_PUB()