
from DataRelated.DataGenerate import ReadRawTrainDataFromFile, ReadRawTestDataFromFile
from Reliability.test_reliability_PUB import  test_Reliability_PUB
from Reliability.test_reliability_moment import test_Reliability_mm

import time
## set the network size
num_node = 10
## set the reliability = 1-\beta
beta_ = 0.1
## set the data size

t1 = time.time()
NumDataSet = [10,25,50,75,100,250,500,750,1000]

num_cov = 2
for num_data in NumDataSet:
    ## Read raw and test data set
    train_data_lst, info = ReadRawTrainDataFromFile(num_node, num_data, num_cov)
    test_data = ReadRawTestDataFromFile(num_node, num_cov)

    test_Reliability_PUB(train_data_lst, test_data, info, num_node, num_data, num_cov, beta_)
    test_Reliability_mm(train_data_lst, test_data, info, num_node, num_data, num_cov)

