from DataRelated.DataGenerate import   GenerateRawDataFileList
from DataRelated.DataGenerateStorm import RawDataGenerateStormFileList
def main():
    ## number of nodes in the network
    num_node_lst = [10]
    ## number of covariates
    num_cov_set = [2]
    max_data_length = 10000

    # given the distribution, generate the synthetic data
    for num_node in num_node_lst:
        for num_cov in num_cov_set:
            GenerateRawDataFileList(num_node, num_cov, max_data_length)
    ## generate synthetic data given NOAA data in case study
    # RawDataGenerateStormFileList(49, train_length= 1, test_length= 1, low_demand=1/3, high_demand=3)

main()