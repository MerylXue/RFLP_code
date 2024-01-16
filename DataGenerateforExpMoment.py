"""
Code for "Hao Shen, Mengying Xue, and Zuojun Max Shen (2023),
Data-driven Reliable Faciltiy Location Problem, Management Science, Forthcoming",

Last Revised on Jan 12, 2024
@author: Mengying Xue
"""

## This code generates data in Lv et al. 2015, with parameter beta, theta
# the lat long of New Orleans 29.9976   90.1774
import math
from DataRelated.DataGenerate import read_from_txt, MapDistance, DistNewOrleans

def IndDisruption(beta_, theta_, dist_NewOrleans):
    num_J = len(dist_NewOrleans)
    DisruptProb = [ beta_*math.exp(-dist_NewOrleans[j]/theta_) for j in range(num_J)]
    return DisruptProb


def DisruptMoment( num_node, beta_, theta_, max_dist):
    file = 'Data/UCFLData%d.txt' % num_node
    mu, f, o, num_I, num_J, coor_I, coor_J = read_from_txt(file)
    f = f + [0]
    dist_NewOrleans = DistNewOrleans(coor_J)
    d0 = MapDistance(coor_I, coor_J)

    dist = [d0[i] + [o[i]] for i in range(num_I)]
    MarginalProb = IndDisruption(beta_, theta_, dist_NewOrleans)

    info = {'dist': dist, 'fixed_cost': f, 'mu': mu, 'num_customer': num_I, 'num_facility': num_J}


    index = [(i,j) for i in range(num_I) for j in range(i, num_J)  if dist[i][j] >= max_dist ]

    SecondMomentProb = []
    IndexPair = []

    for k in index:
        prob = MarginalProb[k[0]] * MarginalProb[k[1]]
        SecondMomentProb.append(prob)
        IndexPair.append([k[0], k[1]])

    return MarginalProb, SecondMomentProb, IndexPair, info
