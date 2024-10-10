"""
Code for "Hao Shen, Mengying Xue, and Zuojun Max Shen (2023),
Data-driven Reliable Faciltiy Location Problem, Management Science, Forthcoming",

Last Revised on Jan 12, 2024
@author: Mengying Xue

"""
## This file is the main function for testing marginal moment and cross moment
# in data with no-covariate case


from DataRelated.DataGenerateNoCov import  ReadRawTrainDataNoCovFromFile, ReadRawTestDataNoCovFromFile
from Reliability.test_reliability_moment import test_Reliability_mm_NoCov


def run_moment():
    num_node = 10
    ## set the reliability = 1-\beta
    beta_lst = [0.1]
    ## set the number of covairates, 2/5


    ## set the data size
    ## test for the no covariates case
    NumDataSet = [10, 25, 50, 75, 100, 250, 500, 750, 1000]
    setting_lst = [(1.6, True)]
    for (mu_coeff, truncate) in setting_lst:
        for num_data in NumDataSet:
            train_data_lst, info = ReadRawTrainDataNoCovFromFile(num_node, mu_coeff, truncate, num_data)
            test_data = ReadRawTestDataNoCovFromFile(num_node, mu_coeff, truncate)
            test_Reliability_mm_NoCov(train_data_lst, test_data, info, num_node, num_data, 1, mu_coeff, truncate)


run_moment()