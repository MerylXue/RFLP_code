from DataRelated.DataGenerate import ReadRawDataSAA
from Reliability.test_reliability_SAA import run_SAA

def test():
    num_node = 10
    num_cov_lst = [2]

    ###%%% Read data from data list
    # NumDataSet = [100,500,1000,5000,10000]
    #
    # for num_data in NumDataSet:
    for num_cov in num_cov_lst:
        real_data, info  = ReadRawDataSAA(num_node, num_cov)
        # train_data, info = ReadRawDataSAA(num_node, num_cov)
        run_SAA(real_data, info, num_node, num_cov)

test()