"""
Code for "Hao Shen, Mengying Xue, and Zuojun Max Shen (2023),
Data-driven Reliable Faciltiy Location Problem, Management Science, Forthcoming",

Last Revised on Jan 12, 2024
@author: Mengying Xue

"""
## This file is the main function for testing marginal moment and cross moment
# in data with covariate and no-covariate case


from DataRelated.DataGenerate import  ReadRawTrainDataFromFile, ReadRawTestDataFromFile
from Reliability.test_reliability_moment import test_Reliability_moment


def run_moment():
    num_node = 10
    num_cov_lst = [2,5]
    NumDataSet = [10,25,50,75,100,250,500, 750, 1000]

    for num_cov in num_cov_lst:
        for num_data in NumDataSet:
            ## Read raw and test data set
            train_data_lst, info = ReadRawTrainDataFromFile(num_node, num_data, num_cov)
            test_data = ReadRawTestDataFromFile(num_node, num_cov)
            ## return the performances of all four methods
            test_Reliability_moment(train_data_lst, test_data, info, num_node, num_data, num_cov)

run_moment()