"""
Code for "Hao Shen, Mengying Xue, and Zuojun Max Shen (2023),
Data-driven Reliable Faciltiy Location Problem, Management Science, Forthcoming",

Last Revised on Jan 12, 2024
@author: Mengying Xue
"""
## For data generation

from DataRelated.DataGenerate import   GenerateRawDataFileList
from DataRelated.DataGenerateStorm import RawDataGenerateStormFileList
from DataRelated.StormDataProcess import DisruptionState
from DataRelated.DataGenerateNoCov import GenerateRawDataNoCovFileList
def main():
    ## number of nodes in the network
    num_node_lst = [10] #20,50 for computational time tests
    ## number of covariates
    max_data_length = 10000
    ##generate the synthetic data with no covariates, demands in high and low
    setting_lst = [(1.6, True), (0.4, True)]
    for num_node in num_node_lst:
        for (mu_coeff, truncate) in setting_lst:
                GenerateRawDataNoCovFileList(num_node, max_data_length, mu_coeff, truncate)
    print("Finish Generating Simulation Data for Section 5.1")

    # generate the synthetic data with covariates
    # number of covariates
    num_cov_set = [2]
    for num_node in num_node_lst:
        for num_cov in num_cov_set:
            GenerateRawDataFileList(num_node, num_cov, max_data_length)
    print("Finish Generating Simulation Data for Section 6.2")
    #
    # generate synthetic data given NOAA data in case study
    RawDataGenerateStormFileList(49, train_length= 1, test_length= 1, low_demand=1/3, high_demand=3)
    print("Finish Generating  Synthetic Data for Case Study in Section 6.3")
    #
    #
    ##  generate the synthetic data for the c
    num_node_lst = [10, 20, 50]  # 20,50 for computational time tests

    max_data_length = 10000
    setting_lst = [(1.6, True)]
    for num_node in num_node_lst:
        for (mu_coeff, truncate) in setting_lst:
            GenerateRawDataNoCovFileList(num_node, max_data_length, mu_coeff, truncate)
    print("Finish Generating Simulation Data for the Efficiency Test in Section 5.2")


    # # functions for generating /Data/Storm/disruption_49.json
    # file_loc = 'Data/data49UFLP.xls'
    # file_storm =  'Data/Storm/StormEvents_details_Begin_1950_End_2021.csv'
    # DisruptionState(file_storm, file_loc)
main()