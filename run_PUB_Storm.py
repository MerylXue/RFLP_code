from DataRelated.DataGenerateStorm import ReadRawDataStormFromFile
from Reliability.test_reliability_PUB import run_Storm_PUB
def run_Storm():
    num_node = 49
    num_cov_lst = [10]

    train_length = 5
    test_length = 1
    beta_ = 0.2
    for num_cov in num_cov_lst:
        #### Generate Storm Sce Data
        train_data_lst, test_data_lst, info = ReadRawDataStormFromFile(num_node, num_cov, train_length, test_length)
        run_Storm_PUB(train_data_lst, test_data_lst, info, num_node, num_cov, beta_,  train_length, test_length)
run_Storm()