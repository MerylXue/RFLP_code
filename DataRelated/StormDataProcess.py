"""
Code for "Hao Shen, Mengying Xue, and Zuojun Max Shen (2023),
Data-driven Reliable Faciltiy Location Problem, Management Science, Forthcoming",

Last Revised on Jan 12, 2024
@author: Mengying Xue

This file is used for processing weather data from NOAA,
i.e., StormEvents_details_Begin_1950_End_2021 in Data/Storm
the synthetic data is simulated in the following way:
In our experiment, each hazard with an estimated property damage over 500,000 dollars
 is recognized as a severe hazard resulting in disruptions.
 We aggregate hazards into two groups: one contains “Marine Thunderstorm Wind”,
 “Thunderstorm Wind”, and “Marine High Wind”, and the other one includes
 all the remainder types of hazards in the data set.

The data is then processed into a json file: disruption_49.json.
In the numerical study, we directly use the json file as the input of storm data
"""

import numpy as np
import xlrd
import math
import pandas as pd
import os
import datetime
from Utility.dist import lat_dis
from Utility.Utils import ConvertNumber
import json
START_YEAR = 1950
END_YEAR = 2021
THRESHOLD_LENGTH = 150
THRESHOLD_PROPERTY = 500000
head_time = datetime.datetime(1950,1, 1)
tail_time = datetime.datetime(2021,12, 31)

class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def read_excel(file):
    wb = xlrd.open_workbook(filename=file)
    sheet1 = wb.sheet_by_index(0)
    # sheet2 = wb.sheet_by_name('data')
    #
    # print(sheet1, sheet2)

    #Get the total number of locatons

    rows = sheet1.row_values(2)  # 获取行内容
    num_J = int(rows[1])
    num_I = num_J


    col_lat = sheet1.col_values(5)
    col_lon = sheet1.col_values(6)
    # the data in Synder and Dustin (2005) is the west longitude, which differs from the NOAA data
    coor_I = [(col_lat[j + 3], -col_lon[j + 3]) for j in range(num_J)]

    return(coor_I)

def location(begin_lat, begin_lon, end_lat, end_lon, coor, num_J):
    disrupt_loc = []
    for j in range(num_J):
        d1 = lat_dis((begin_lat, begin_lon), coor[j])
        d2 = lat_dis((end_lat, end_lon), coor[j])
        if d1 < THRESHOLD_LENGTH or d2 < THRESHOLD_LENGTH:
            disrupt_loc.append(j)

    return disrupt_loc

def LoadDatafromCSV(data, coor):
    # csvFile = open(file, "r")
    num_J = len(coor)
    # print(data['DAMAGE_PROPERTY'])

    event_type = list(set(data['EVENT_TYPE']))
    num_events = len(event_type)
    num_period = (END_YEAR - START_YEAR + 1) * 4
    disruption_stats = pd.DataFrame(columns = ['event','start_time','end_time']+[j for j in range(num_J)])
    disruption_stats['event'] = [event for t in range(num_period) for event in event_type]

    idx = 0


    for event_id in range(num_events):
        data_event = data[data['EVENT_TYPE'] == event_type[event_id]].reset_index(drop = True)
        disruptions = np.zeros((num_period, num_J))
        if len(data_event) > 0:
            for i in range(len(data_event)):
                begin_year = int(data_event['BEGIN_YEARMONTH'][i] /100)
                begin_month = int(data_event['BEGIN_YEARMONTH'][i] - 100 * begin_year)
                end_year = int(data_event['END_YEARMONTH'][i] / 100)
                end_month = int(data_event['END_YEARMONTH'][i] - 100 * end_year)

                start_time = (begin_year - START_YEAR) * 4 + math.ceil(begin_month/3) - 1
                end_time = (end_year - START_YEAR) * 4 + math.ceil(end_month/3) - 1

                begin_lat = data_event['BEGIN_LAT'][i]
                begin_lon = data_event['BEGIN_LON'][i]
                end_lat = data_event['END_LAT'][i]
                end_lon = data_event['END_LON'][i]

                # if no exact information about location
                if np.isnan(begin_lat) and np.isnan(begin_lon) and np.isnan(end_lat) and np.isnan(end_lon):
                    continue
                disrupt_loc = location(begin_lat, begin_lon, end_lat, end_lon, coor, num_J)


                damage = ConvertNumber(data['DAMAGE_PROPERTY'][i])
                if damage > THRESHOLD_PROPERTY:
                    for j in disrupt_loc:
                        for t in range(start_time, end_time + 1):
                            # print(t, j)
                            disruptions[t][j] += 1
            if np.sum(disruptions) > 0:
                disruption_stats.update({event_type[event_id]: disruptions})
            # print(disruption_stats)
    return disruption_stats

def file_name_list():
    filePath = 'Data/Storm'
    fileNameList = os.listdir(filePath)
    return fileNameList

def MergeStormFiles():

    data = []
    fileNameList = file_name_list()
    for i in range(len(fileNameList)):
        path = 'Data/Storm/'
        if fileNameList[i][-1] == 'v':
            file = path + fileNameList[i]
            if i == 0:
                data = pd.read_csv(file, low_memory=False)
            else:
                data = data.append(pd.read_csv(file, low_memory=False), ignore_index=True)
                print("Data length %d" % len(data))
        else:
            continue

    data = data[(data['BEGIN_LAT'])]
    data.to_csv('Data/Storm/StormEvents_details_Begin_%d_End_%d.csv' % (START_YEAR, END_YEAR), index=False)

def DisruptionState(file_storm, file_loc):
    data=  pd.read_csv(file_storm, low_memory=False)

    coor = read_excel(file_loc)
    num_J = len(coor)

    disruption_stats = LoadDatafromCSV(data, coor)
    json_str = json.dumps(disruption_stats, cls=NumpyArrayEncoder)
    print(json_str)
    with open('Data/Storm/disruption_%d.json'%num_J,'w') as file:
        file.write(json_str)



      #
    # # disruption_stats = disruption_stats/num_period
    # output_file = 'disruption_%d.csv'%num_J
    # with open(output_file, mode = 'w') as output:
    #     for t in range(num_period):
    #         out = str(list(disruption_stats[t]))
    #         out = out.replace("[","")
    #         out = out.replace("]","")
    #         out_line = [out]
    #         print(out_line)
    #         output.writelines(','.join(out_line) + '\n')
    # print(disruption_stats)




