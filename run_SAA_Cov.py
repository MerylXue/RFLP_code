from DataRelated.DataGenerate import ReadRawDataSAA
from Reliability.test_reliability_SAA import run_SAA_cov

def test():
    num_node = 10
    num_cov = 2

    # for num_data in NumDataSet:
    real_data, info  = ReadRawDataSAA(num_node, num_cov)
    # train_data, info = ReadRawDataSAA(num_node, num_cov)
    run_SAA_cov(real_data, info, num_node, num_cov)

test()