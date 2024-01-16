"""
Code for "Hao Shen, Mengying Xue, and Zuojun Max Shen (2023),
Data-driven Reliable Faciltiy Location Problem, Management Science, Forthcoming",

Last Revised on Jan 12, 2024
@author: Mengying Xue

"""
from SampleAverage.SAA import  SAA1
## return the optimization results for sample average approximation
def run_SAA(rawdata_train, info,  num_node, num_cov):
    outputfile = open(
        'result/SAA_node%d_cov%d.csv' % ( num_node, num_cov), 'w')
    outline1 = [ 'obj_value',   'Num_loc', 'sol_time', 'str_loc','gap']
    outputfile.writelines(','.join(outline1) + '\n')
    loc_chosen, obj_value, t, gap = SAA1(rawdata_train, info)
    str_loc = ";".join([str(x) for x in loc_chosen])
    outline =  ["%.2f" % obj_value] + ['%d' % len(loc_chosen)] + ['%.2f' % t] +['%s'%str_loc] + ['%f' % gap]
    outputfile.writelines(','.join(outline) + '\n')
    outputfile.close()
    # print("Run SAA Finished")