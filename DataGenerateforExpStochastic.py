"""
Code for "Hao Shen, Mengying Xue, and Zuojun Max Shen (2023),
Data-driven Reliable Faciltiy Location Problem, Management Science, Forthcoming",

Last Revised on Jan 12, 2024
@author: Mengying Xue
"""


## This code generates data in Aboolian et al. 2013, with parameter alpha
# the lat long of New Orleans 29.9976   90.1774
import math
from DataRelated.DataGenerate import read_from_txt
from DataRelated.DataGenerate import MapDistance, DistNewOrleans

def IndDisruption(num_node, alpha_):
    file = 'Data/UCFLData%d.txt' % num_node
    mu, f, o, num_I, num_J, coor_I, coor_J = read_from_txt(file)

    f = f + [0]
    dist_NewOrleans = DistNewOrleans(coor_J)

    DisruptProb = [0.01 + float((int(0.1*alpha_*math.exp(-dist_NewOrleans[j]/400)* 1E+6))/1E+6) for j in range(num_J)]

    d0 = MapDistance(coor_I, coor_J)
    dist = [d0[i] + [o[i]] for i in range(num_I)]

    info = {'dist': dist, 'fixed_cost': f, 'mu': mu, 'num_customer': num_I, 'num_facility': num_J}

    return DisruptProb, info