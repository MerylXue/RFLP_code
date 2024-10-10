from DataRelated.DataGenerate import ReadRawDataSAA
from DataRelated.DataGenerateNoCov import ReadRawDataSAANoCov
from Reliability.test_reliability_SAA import run_SAA_NoCov

def test():
    num_node = 10

    ###%%% Read data from data list
    # mu_coeff = 0.4
    mu_coeff = 1.6
    truncate = True


    real_data, info  = ReadRawDataSAANoCov(num_node, mu_coeff, truncate)
    run_SAA_NoCov(real_data, info, num_node, mu_coeff, truncate)

test()