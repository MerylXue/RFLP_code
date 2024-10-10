from DataRelated.DataGenerateNoCov import ReadRawTrainDataNoCovFromFile, ReadRawTestDataNoCovFromFile
from Reliability.test_reliability_PUBNoCov import  test_Reliability_PUB_NoCov
from Reliability.test_reliability_Wasserstein import test_Reliability_Wasserstein

import time
## set the network size
num_node = 10
## set the reliability = 1-\beta
beta_ = 0.1
## set the number of covairates, 2/5


## set the data size
## test for the no covariates case

t1 = time.time()
NumDataSet = [10,25,50,75,100,250,500,750,1000]
mu_coeff = 1.6
truncate = 1
for num_data in NumDataSet:
    train_data_lst, info = ReadRawTrainDataNoCovFromFile(num_node, mu_coeff, truncate, num_data)
    test_data = ReadRawTestDataNoCovFromFile(num_node, mu_coeff, truncate)

    test_Reliability_PUB_NoCov(train_data_lst, test_data, info, num_node, num_data, 1, beta_, mu_coeff, truncate)
    test_Reliability_Wasserstein(train_data_lst, test_data, info, num_node, num_data, 1, beta_, mu_coeff, truncate)

t2 = time.time()

print("total time is %f"%(t2-t1))