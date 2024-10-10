"""
Code for "Hao Shen, Mengying Xue, and Zuojun Max Shen (2023),
Data-driven Reliable Faciltiy Location Problem, Management Science, Forthcoming",

Last Revised on Jan 12, 2024
@author: Mengying Xue

"""

## this file is main file for generating the performance in Section 5.1

from DataRelated.DataGenerateNoCov import ReadRawTrainDataNoCovFromFile, ReadRawTestDataNoCovFromFile
from Reliability.test_reliability_PUBNoCov import  test_Reliability_PUB_NoCov
def run_PUB():

    ## set the network size
    num_node = 10
    ## set the reliability = 1-\beta
    beta_lst = [0.1]
    ## set the number of covairates, 2/5


    ## set the data size
    ## test for the no covariates case
    # NumDataSet = [10]
    NumDataSet = [250]
    setting_lst = [(1.6, True)]
    for (mu_coeff, truncate) in setting_lst:
        for num_data in NumDataSet:
            for beta_ in beta_lst:
                train_data_lst, info = ReadRawTrainDataNoCovFromFile(num_node, mu_coeff, truncate, num_data)
                test_data = ReadRawTestDataNoCovFromFile(num_node, mu_coeff, truncate)
            ## for covariate case
                test_Reliability_PUB_NoCov(train_data_lst, test_data, info, num_node, num_data, 1, beta_, mu_coeff, truncate)
run_PUB()