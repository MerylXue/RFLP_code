"""
Code for "Hao Shen, Mengying Xue, and Zuojun Max Shen (2023),
Data-driven Reliable Faciltiy Location Problem, Management Science, Forthcoming",

Last Revised on Jan 12, 2024
@author: Mengying Xue

"""
## This file is the main function for testing wasserstein based method.
## In the manuscript, we only test Wasserstein based method in no-covariate case



from DataRelated.DataGenerateNoCov import  ReadRawTrainDataNoCovFromFile, ReadRawTestDataNoCovFromFile
from Reliability.test_reliability_Wasserstein import test_Reliability_Wasserstein
def run_PUB():
    num_node = 10
    beta_lst = [0.1]
    num_cov_set = [2]
    NumDataSet = [750]
    ## beta = 0.25, data = 100

    setting_lst = [(1.6, True)]
    for (mu_coeff, truncate) in setting_lst:
        for num_data in NumDataSet:
            for beta_ in beta_lst:
                train_data_lst, info = ReadRawTrainDataNoCovFromFile(num_node, mu_coeff, truncate, num_data)
                test_data = ReadRawTestDataNoCovFromFile(num_node, mu_coeff, truncate)
                test_Reliability_Wasserstein(train_data_lst, test_data, info, num_node, num_data, 1, beta_, mu_coeff, truncate)
run_PUB()