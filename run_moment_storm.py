"""
Code for "Hao Shen, Mengying Xue, and Zuojun Max Shen (2023),
Data-driven Reliable Faciltiy Location Problem, Management Science, Forthcoming",

Last Revised on Jan 12, 2024
@author: Mengying Xue

"""
## This file is the main function for testing marginal moment and cross moment with covariates in NOAA data set


from DataRelated.DataGenerateStorm import ReadRawDataStormFromFile
from Reliability.test_reliability_moment import run_Storm_mm
def run_Storm():
    num_node = 49
    num_cov = 2

    train_length = 1
    test_length = 1
    beta_ = 0.2
    ## Read case study data set
    train_data_lst, test_data_lst, info = ReadRawDataStormFromFile(num_node, num_cov, train_length, test_length)
    run_Storm_mm(train_data_lst, test_data_lst, info, num_node, num_cov, beta_,  train_length, test_length)
run_Storm()