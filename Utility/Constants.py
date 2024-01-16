"""
Code for "Hao Shen, Mengying Xue, and Zuojun Max Shen (2023),
Data-driven Reliable Faciltiy Location Problem, Management Science, Forthcoming",

Last Revised on Jan 12, 2024
@author: Mengying Xue

"""

## This file gives the constants used in the code
eps = .01

demand_coeff = 2

NumTestData = 10000
## reliability
## the right end for binary search
Right = 1
# the convergence for binary search
converge_diff = 1E-3
# number of iteration for calculating reliability
Num_reliability = 50
Num_real = 1000000
MAXNUMBER = 1E9
# number of simulate for comparing the reliability
NumSimulate = 10
## search step in the binary search of ambiguity size
search_step = 8

## time periods used in case using weather data
T_period = int((2019 - 1950) / 5) + 1
T_threshold = [[0,0] for i in range(T_period)]

START_YEAR = 1955
END_YEAR = 2019

EpsSet = [i * 1E-1 for i in range(2,10,2)] + [i * 1E-2 for i in range(2,10,2)] + [i * 1E-3 for i in range(1,10)] \
         + [i * 1E-4 for i in range(1,10)]
NumCrossValidation = 5