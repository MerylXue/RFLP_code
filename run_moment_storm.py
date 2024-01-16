"""
Code for "Hao Shen, Mengying Xue, and Zuojun Max Shen (2023),
Data-driven Reliable Faciltiy Location Problem, Management Science, Forthcoming",

Last Revised on Jan 12, 2024
@author: Mengying Xue

"""
## This file is the main function for testing marginal moment and cross moment with covariates in NOAA data set


from DataRelated.DataGenerateStorm import ReadRawDataStormFromFile
from Reliability.test_reliability_moment import run_Storm_moment
def run_Storm():
    num_node = 49
    num_cov_lst = [10]

    train_length = 5
    test_length = 1
    beta_ = 0.2
    for num_cov in num_cov_lst:
        ## Read NOAA data set
        train_data_lst, test_data_lst, info = ReadRawDataStormFromFile(num_node, num_cov, train_length, test_length)
        run_Storm_moment(train_data_lst, test_data_lst, info, num_node, num_cov, beta_,  train_length, test_length)
run_Storm()