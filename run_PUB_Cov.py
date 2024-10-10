"""
Code for "Hao Shen, Mengying Xue, and Zuojun Max Shen (2023),
Data-driven Reliable Faciltiy Location Problem, Management Science, Forthcoming",

Last Revised on Jan 12, 2024
@author: Mengying Xue

"""

## this file is main file for generating the performance in Section 5.1

from DataRelated.DataGenerate import  ReadRawTrainDataFromFile, ReadRawTestDataFromFile
from Reliability.test_reliability_PUB import test_Reliability_PUB
def run_PUB():

    ## set the network size
    num_node = 10
    ## num_node = 20,50
    ## set the reliability = 1-\beta
    beta_lst = [0.1]
    ## set the number of covairates, 2/5
    num_cov_set = [2]

    ## set the data size
    # NumDataSet = [75]
    NumDataSet = [10,25,50,75,100,250,500,750,1000]
    ## For computational time test: NumDataSet = [100,500,1000,5000,10000]

    for num_data in NumDataSet:
        for num_cov in num_cov_set:
            for beta_ in beta_lst:
                train_data_lst, info = ReadRawTrainDataFromFile(num_node, num_data, num_cov)
                test_data = ReadRawTestDataFromFile(num_node, num_cov)
                ## for covariate data
                test_Reliability_PUB(train_data_lst, test_data, info, num_node, num_data, num_cov, beta_)
                ## for covariate case
run_PUB()