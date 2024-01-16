"""
Code for "Hao Shen, Mengying Xue, and Zuojun Max Shen (2023),
Data-driven Reliable Faciltiy Location Problem, Management Science, Forthcoming",

Last Revised on Jan 12, 2024
@author: Mengying Xue

"""

## This file defines the function on calculating the distance between two locations

import math

def dist(coor1, coor2):
	return math.sqrt((coor1[0] - coor2[0])**2 + (coor1[1] - coor2[1])**2)

def deg2rad(deg):
    return deg * math.pi/180

def rad2deg(rad):
    return rad * 180/math.pi

# return the distance between two given locations
def lat_dis(coor1, coor2):
    lat1 = (math.pi/180)*coor1[0]
    lat2 = (math.pi/180)*coor2[0]
    lon1 = (math.pi/180)*coor1[1]
    lon2= (math.pi/180)*coor2[1]

    #因此AB两点的球面距离为:{arccos[sinb*siny+cosb*cosy*cos(a-x)]}*R
    #地球半径 km
    # R = 6378;
    # R = 3958.7613 #MILES
    # print(math.sin(lat1)*math.sin(lat2)+ math.cos(lat1)*math.cos(lat2)*math.cos(lon2-lon1))
    if ( math.fabs(lat1 - lat2) + math.fabs(lon1-lon2)) < 1e-4:
        return 0
    d = math.acos(math.sin(lat1)*math.sin(lat2)+ math.cos(lat1)*math.cos(lat2)*math.cos(lon2-lon1))

    d = round(float(int(d * 3958.7613 * 1e2)/1e2),2)
    return d

# return the distance between the given location and NewOrleans, the center of storm
def Distance_NewOrleans(coor):

    coor_NewOrleans = [30.07, 89.93]
    # in miles
    d = lat_dis(coor, coor_NewOrleans)
    return d
