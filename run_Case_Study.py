from DataRelated.DataGenerateStorm import ReadRawDataStormFromFile
from Reliability.test_reliability_PUB import run_Storm_PUB
from Reliability.test_reliability_moment import run_Storm_mm

num_node = 49
num_cov = 2

train_length = 1
test_length = 1
beta_ = 0.2
#### Read  case study data set
train_data_lst, test_data_lst, info = ReadRawDataStormFromFile(num_node, num_cov, train_length, test_length)
run_Storm_PUB(train_data_lst, test_data_lst, info, num_node, num_cov, beta_,  train_length, test_length)
run_Storm_mm(train_data_lst, test_data_lst, info, num_node, num_cov, beta_,  train_length, test_length)