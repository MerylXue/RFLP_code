# RFLP_code

## Overview

This repository contains numerical implementation for reliable facility location problem (RFLP) in the paper [Data-driven Reliable Faciltiy Location Problem](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4525872). 

## Folders

- `Data/`: Network data from Synder and Daskin (2005) [source]([Larry Snyder &raquo; Data Sets for &#8220;Reliability Models for Facility Location: The Expected Failure Cost Case&#8221;](https://coral.ise.lehigh.edu/larry/research/data-sets-for-reliability-models-for-facility-location-the-expected-failure-cost-case/)), processed weather data from NOAA from 1950 to 2021 in json file, [source]([Storm Events Database | National Centers for Environmental Information](https://www.ncdc.noaa.gov/stormevents/ftp.jsp)[Storm Events Database | National Centers for Environmental Information](https://www.ncdc.noaa.gov/stormevents/ftp.jsp); folders for synthetic data
- `DataRelated/`: Python code for preprocessing raw data and generating synthetic data used in numerical studies.
- `MomentBased/`: Python code for marginal-moment method in [Lu et al. (2015)]([Reliable Facility Location Design Under Uncertain Correlated Disruptions | Manufacturing & Service Operations Management](https://pubsonline.informs.org/doi/abs/10.1287/msom.2015.0541)) and cross moment method in [Li et al. (2022)]([A General Model and Efficient Algorithms for Reliable Facility Location Problem Under Uncertain Disruptions | INFORMS Journal on Computing](https://pubsonline.informs.org/doi/abs/10.1287/ijoc.2021.1063))
- `results/`: optimization results
- `Wasserstein/`: Python code for optimizing the RFLP using type-infinity wasserstein DRO in [Xie  (2020)]([Tractable reformulations of two-stage distributionally robust linear programs over the type-∞ Wasserstein ball - ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0167637720300857)[Tractable reformulations of two-stage distributionally robust linear programs over the type-∞ Wasserstein ball - ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0167637720300857))
- `PUB/`: Python code for RFLP using PUB estimator in [Data-driven Reliable Faciltiy Location Problem](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4525872).
- `SampleAverage/`: Python code for RFLP using sample average approximation method.
- `Reliability/`: Python codes for testing reliability and generating average performances of RFLP using PUB estimator, Wasserstein DRO, and moment based methods.
- `Utility/`: Python code for utility functions and constants used in the experiments
- `Plot/`: Python code for the plots in the paper [Data-driven Reliable Faciltiy Location Problem](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4525872).
