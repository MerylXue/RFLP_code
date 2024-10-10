# RFLP_code

## Overview

**Electronic companion of the paper:** *[Data-Driven Reliable Facility Location Design](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4525872)* by

- Hao Shen: School of Business, Renmin University of China,  Beijing, China,  [shenhao@rmbs.ruc.edu.cn](mailto:shenhao@rmbs.ruc.edu.cn)
- Mengying Xue: International Institute of Finance, School of Management, University of Science and Technology of Chinaa, Hefei, Anhui, China, [xmy2021@ustc.edu.cn](mailto:xmy2021@ustc.edu.cn)
- Zuo-Jun Max Shen: Faculty of Engineering;  Faculty of Business and Economics, The University of Hong Kong, Hong Kong, China [maxshen@hku.hk](mailto:maxshen@hku.hk)

This code was tested on:

- Ubuntu server equipped with 20 processors and 40G RAM.
- Gurobi 9.5 with Python API  

## Folders and Related Scripts

- `Data/`: Network data from  [Synder and Daskin (2005)]([Larry Snyder &raquo; Data Sets for &#8220;Reliability Models for Facility Location: The Expected Failure Cost Case&#8221;](https://coral.ise.lehigh.edu/larry/research/data-sets-for-reliability-models-for-facility-location-the-expected-failure-cost-case/)), processed weather data from NOAA from 1950 to 2021 in json file, [source]([Storm Events Database | National Centers for Environmental Information](https://www.ncdc.noaa.gov/stormevents/ftp.jsp)[Storm Events Database | National Centers for Environmental Information](https://www.ncdc.noaa.gov/stormevents/ftp.jsp); folders for synthetic data
  - `RawDataNoCov/`: store the generated synthetic data based on randomly generated distributions in the no-covariate case. The data format is .csv.
  - `RawDataNoCovFileName/`: store the name lists for the synthetic data with no covariates. The names are recorded in .txt files.
  - `RawData/`: store the generated synthetic data based on randomly generated distributions in the multi-covariate case
  - `RawDataFileName/`: store the name lists for the synthetic data with covariates. The names are recorded in .txt files.
  - `RawDataDistribution/`: store the randomly generated distributions on supplies and demands. The data format is .pkl. For the no covariate tests, the file is with the name "Node_%d-mu_%f.pkl", with the number of nodes in the network and the demand means (High-1.6 or Low-0.4). For the tests with multi-covaraites, the file is in the name "Node_%d-cov_%d.pkl", with the number of nodes and the number of covariates. 
  - `RawDataStorm/`: store the generated data sets for the case study in Section 6. Each generated file records the states of each of 49 locations in 12 months over a year, from 1997 to 2021.
  - `RawDataStormFileName/`: store the name lists for the data sets for case study.
  - `Storm/`:
    - `disruption_49.json`: synthetic data file for the network with 49 nodes with disruption states using the data from NOAA. We refer readers to Section 6 in the manuscript to see the data processing of the weather data. Each row represents one historical record of weather hazard, and each column represents the state (disruption or not) of the 49 locations on the network. To generation this file, run `DisruptionState()` in `data_generate.py`.
    - `StormEvents_details_Begin_1950_End_2021.csv`: the aggregated data of the historical records downloaded from NOAA. Used for generating `disruption_49.json`.  
  - `dataXXXUFLP.xls`: XX in [49,50,100,150], location data in [Synder and Daskin (2005)]([Larry Snyder » Data Sets for “Reliability Models for Facility Location: The Expected Failure Cost Case”](https://coral.ise.lehigh.edu/larry/research/data-sets-for-reliability-models-for-facility-location-the-expected-failure-cost-case/))
  - `UCFLDataXX.txt`: XX in [10,20,50,75,100,150], location data in Synder and Daskin (2005) and later processed by [Li et al. (2022)]([A General Model and Efficient Algorithms for Reliable Facility Location Problem Under Uncertain Disruptions | INFORMS Journal on Computing](https://pubsonline.informs.org/doi/abs/10.1287/ijoc.2021.1063))
- `DataRelated/`: Python code for preprocessing raw data and generating synthetic data used in numerical studies.
  - ``DataGenerateNoCov.py` :generate synthetic data for the numerical experiments in Section 5.
  - `DataGenerate.py`: load the information from Synder and Daskin's data set, and generate synthetic data in the multi-covarate case for the numerical experiments in Section 6.2.
  - `DataGenerateStorm.py`: generate the monthly location states data based on the weather data from NOAA for the test in Section 6.3. 
  - `DataProcess.py`: used for the data processing before the optimization in our PUB estimator. 
  - `StormDataProcess.py`: process the original weather data from NOAA. See the details in Section 6 in the paper. 
- `MomentBased/`: Python code for marginal-moment method in [Lu et al. (2015)]([Reliable Facility Location Design Under Uncertain Correlated Disruptions | Manufacturing & Service Operations Management](https://pubsonline.informs.org/doi/abs/10.1287/msom.2015.0541)) and cross moment method in [Li et al. (2022)]([A General Model and Efficient Algorithms for Reliable Facility Location Problem Under Uncertain Disruptions | INFORMS Journal on Computing](https://pubsonline.informs.org/doi/abs/10.1287/ijoc.2021.1063))
  - `CrossMomentFunc.py`: functions for cross moment method
  - `SUPCrossMoment.py`: the coonstraint-generation optimizaiton function for cross moment method. This original method was revised by us.
  - `MarginalMoment.py`: functions for marginal moment method, including the constraint-generation optimziation function,
  - `MomentData.py`: functions on obtaining the moment information from the data.  
- `results/`: forlder for average performances of different methods
  - `/reliability/`: detailed results for each data set
  - `/Storm/`: results for the case study
- `Wasserstein/`: Python code for optimizing the RFLP using type-$\infty$ wasserstein DRO in [Xie  (2020)]([Tractable reformulations of two-stage distributionally robust linear programs over the type-∞ Wasserstein ball - ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0167637720300857)[Tractable reformulations of two-stage distributionally robust linear programs over the type-∞ Wasserstein ball - ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0167637720300857))
  - `Wasserstein.py `
- `PUB/`: Python code for RFLP using PUB estimator proposed by [Data-driven Reliable Faciltiy Location Problem](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4525872).
  -  `SUPDataDriven.py`: constraint-generation algorithm
  - `SUPFunc.py`: functions on obtaining the cost, worst-case distributions
- `SampleAverage/`: Python code for RFLP using sample average approximation method.
  - `SAA.py  `: optimizaiton function for SAA
  - `SAAFunc.py` functions on obtaining the cost in SAA
- `Reliability/`: Python codes for testing reliability and generating average performances of RFLP using PUB estimator, Wasserstein DRO, and moment based methods.
  - `Reliability_PUB.py `:functions on finding the conservative paramters for PUB estimators in in the paper [Data-driven Reliable Faciltiy Location Design](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4525872).
  - `Reliability_Wass.py`: functions on finding the conservative paramters for type-$\infty$ Wasserstein DRO in [Xie (2020)]([Tractable reformulations of two-stage distributionally robust linear programs over the type-∞ Wasserstein ball - ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0167637720300857)[Tractable reformulations of two-stage distributionally robust linear programs over the type-∞ Wasserstein ball - ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0167637720300857)). 
  - `test_reliability_moment.py`: generate the performances of four moment-based methods, including marginal moment with no covariates and with covariates, and cross-moment method with no covariates and with covariates,  in the numerical experiments, including the objectives, out-of-sample costs, reliabilty, optimality gap, computational time. 
  - `test_reliability_PUBNoCov.py`: generate the performances for the PUB estimator in the no-covariate case.
  - `test_reliability_PUB.py`: generate the performances for the PUB estimator in the multi-covariate case. 
  - `test_reliability_SAA.py`: generate the performances for the sample average approximation method.
  - `test_reliability_Wass.py`: generate performances for the type-$\infty$  average approximation method.
- `Utility/`: Python code for utility functions and constants used in the experiments
  - `Constants.py`: constants used in the experiments
  - `dist.py`: functions on calculating the distance between two locations
  - `Utils.py`: utility functions
- `Plot/`: Python code for the plots in the paper [Data-driven Reliable Faciltiy Location Design](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4525872).
  - `plot.py   `

## Scripts and Modules

 Run each of the following scripts, and the result file is in the format of `.csv` and  will be stored in the folder `/results/`

- `data_generate.py`: generate synthetic data and case study data
- `run_PUB_NoCov.py` generate performances for PUB estimators in data with no covaraite (results for Section 5.1 and 5.2)
- `run_PUB_Cov.py`:generate performances for PUB estimators in data with covaraites (results for Section 6.2)
- `run_moment_NoCov.py`: generate test results for marginal momemt and cross moment methods in the data with no covariate (results for Section 5.1 and 5.2)
- `run_moment_Cov.py`: generate test results for the marginal momemt with no cov and cov,  and cross moment method with no cov and cov in the data with multi-covaraites (results for Section 6.2)
- `run_wass.py`: generate test results   type-$\infty$  wasserstein DRO method. 
- `run_PUB_Storm.py`: generate the performance of PUB estimator with covariates in the case study using the weather data from NOAA (results in Section 6.3)
- `run_moment_storm.py`: generate the performance of marginal and cross moment based methods with covariates in the case study using the weather data from NOAA (results in Section 6.3)
- `run_SAA_NoCov.py`: generate the approximated true optimum for the tests in Section 5.1.
- `run_SAA_Cov.py`: generate the approximated true optimum for the tests in Section 6.2.
- `plot_fig.py`: generate plots comparing different methods in the paper

# Tests

Settings in `Utility/Constants.py`: NumSimulate = 100, Num_reliability = 50, Num_real = 1000000

## Numerical Tests in Section 5.1

***Description***:  This experiment is designed to compare the in-sample, out-of-sample performances, and the reliability of our PUB estimator compared with the type-$\infty$ Wasserstein DRO, marginal moment method, and the cross moment method.

**Data**: This experiment uses a network with 10 nodes. The location information is in `Data/UCFLData10.txt`. Run function `GenerateRawDataNoCovFileList()` in `data_generate.py`, with the setting *num_node_lst = [10]setting_lst = [(1.6, True), (0.4, True)], mu_coeff = 1.6/0.4, truncate = True*, where *mu_coeff=1.6* denotes the high-demand case (H), and *mu_coeff=0.4* denotes the low-demand case (L). The generated data will be stored in `/Data/RawDataNoCov` folders. Results for each data samples are in the name `%s_data%d_node%d_cov1_beta%f_Mu_%f_Truncate_%d.csv`(method name, number of nodes, number of data sets, reliability level, mu_coeff, truncate) in folder `/result/Reliability/`

**Experiments**: 

- Run `run_moment_NoCov.py` for the results of marginal moment and cross-moment methods;

-  Run`run_wass.py` for the results of the type-$\infty$ Wasserstein DRO method; 

- Run `run_PUB_NoCov.py` for the results of PUB estimator in our paper. 

- The settings are: *num_node = 10, mu_coeff = 1.6/0.4, truncate = True, num_data=[10,25,50,100,250,500,750,1000]* 

**Results**

- The average performance results are in the name *Reliability_%s_Node%d_Data%d_Cov%d_Beta%f_Mu_%f_Truncate_%d.csv* (method name, number of nodes, number of data sets, number of covaraites=1, reliability $\beta$ , mu_coeff, truncate) in the folder `/result`, 

- Method name abbreviations:
  
  - `KolNoCov` PUB estimator with no covaraite
  
  - `Wass`: type-$\infty$ Wasserstein DRO
  
  - `moment`: summary files for moment-based methods. `CM_nocov` stands for cross moment method, `MM_nocov` stands for marginal moment method.
  
  - `MaringalMoment_nocov`: marginal moment method
  
  - `CrossMoment_nocov`: cross-moment method

**Benchmark**

The experiments uses a large data set with 10000 data points and sample average method to approximate the true optimum value. Run `run_SAA_NoCov.py` to generate the true optimum. 

## Numerical Tests in Section 5.2

***Description***: This experiment is designed to compare the computational efficiency of our PUB estimator compared with the type-\infty Wasserstein DRO, marginal moment method, and the cross moment method.

**Data**: This experiment uses a network with 10, 20 and 50 nodes. The location information is in `Data/UCFLData10.txt`,`Data/UCFLData20.txt` , and `Data/UCFLData50.txt` Run function `GenerateRawDataNoCovFileList()` in `data_generate.py`, with the setting *num_node_lst = [10,20,50], mu_coeff = 1.6, truncate = True*. The generated data will be stored in `/Data/RawData` folders.

**Experiments** : follow the description on **Numerical Tests in Section 5.1**. The experiments and the reults are shown in a similar way. 

## Numerical Tests in Section 6.2

***Description***: This experiment is designed to compare the computational efficiency of our PUB estimator compared with th marginal moment method, and the cross moment method when there exists discrete covaraite information. 

**Data**: This experiment uses a network with 10 nodes, 2 covaraites. The location information is in `Data/UCFLData10.txt`. Run function `GenerateRawDataFileList()` in `data_generate.py`, with the setting *num_node_lst = [10], num_cov_set = [2]*. The generated data will be stored in `/Data/RawData` folders. 

**Experiments**:

- Run `run_moment_Cov.py` for the results of marginal moment and cross-moment methods with covariates. The function will return the results of margimal moment with covariates, marginal moment with no covariates, cross momen with covariates, cross moment with no covariates. The method with no covariate means that we ignore the covariate information and run the original method (marginal moment or cross moment). 

- Run `run_PUB_Cov.py` for the results of PUB estimator in our paper.

- The settings are: *num_node = 10, num_cov =2, num_data=[10,25,50,100,250,500,750,1000]*

**Results**

- The average performance results are in the name *Reliability_%s_Node%d_Data%d_Cov%d_Beta%f.csv* (method name, number of nodes, number of data sets, number of covaraites, reliability \beta ) in the folder `/result`. Results for each data samples are in the name `%s_data%d_node%d_cov%d.csv`(method name, number of nodes, number of data sets, number of covaraites, ) in folder `/result/Reliability/`
- Method name abbreviations:
  -  `Kol` or `Kol_cov`: PUB estimator with covaraites 
  - `moment`: summary files for moment-based methods. `CM_nocov` stands for cross moment method, `MM_nocov` stands for marginal moment method, `CM_cov` stands for cross moment method with covariates, and `MM_cov` stands for marginal moment method with covaraites,
  - `MaringalMoment_cov`: marginal moment method with covaraites
  - `CrossMoment_cov`: cross-moment method with covaraites

**Benchmark**

The experiments uses a large data set with 10000 data points and sample average method to approximate the true optimum value. Run `run_SAA_Cov.py` to generate the true optimum.

# Numerical Tests in Section 6.3

***Description***: This experiment is designed to use the data generated based on the weather hazards in USA from 1996 to 2021. We refer readers to Section 6.3 in the paper for more detailed description on data generation. 

**Data**: The original historical data has been processed in a json file, see `/Data/Storm/disruption_49.json'`. Data in the json file is in the dictionary form, with {`hazard type`: `disruption state matrix`}, where the row for disruption state matrix is a month in the year 1997-2021, and the column is the location index of the 49 locations. Each value in the `disruption state matrix` denotes the number of disruptions in the location in one month. Run function `RawDataGenerateStormFileList()` in the file `data_generate.py` to generate the disruption state for each location in each month, and a data file represents the record of 12 months in a year. The data will be put in the folder `/Data/RawDataStorm`, with the name `Train-TrainLength_1-TestLength_1-Node_49-Cov_2-Year_2020.csv` (training data in year 2020) or `Test-TrainLength_1-TestLength_1-Node_49-Cov_2-Year_2020.csv`(test data for 2020). We use the data year t for training, and year t+1 for testing, with t = 1996-2020. 

**Experiments**

Set *$\beta$=0.2 , num_cov = 2, num_node = 49, train_length=test_length=1*.

- Run `run_moment_storm.py` for the results of marginal moment and cross-moment methods with covariates.  

- Run `run_PUB_Storm.py` for the results of PUB estimator in our paper.

**Results**

Results will be put in the folder `/result/Storm/`, with the name *%s-Node_%d_Cov_%d-Beta_%f.csv*(method name, number of nodes=49, number of covariates=2, beta = 0.2). `Kol_cov` represents our PUB estimator with covariates. `Moment` is the moment-based methods, with  `CM_cov` representing the cross-moment method with covariates, `MM_cov` representing the marginal moment with covariates. 
