from SampleAverage.SAA import  SAA1
def run_SAA_NoCov(rawdata_train, info,  num_node, mu_coeff, truncate):
    outputfile = open(
        'result/SAA_node%d_Mu_%f_Truncate_%d.csv' % ( num_node, mu_coeff, truncate), 'w')
    print(len(rawdata_train))
    outline1 = [ 'obj_value',   'Num_loc', 'sol_time', 'str_loc','gap']
    outputfile.writelines(','.join(outline1) + '\n')
    loc_chosen, obj_value, t, gap = SAA1(rawdata_train, info)
    str_loc = ";".join([str(x) for x in loc_chosen])
    # saa = SAACost(loc_chosen, info['fixed_cost'], info['dist'], info['num_customer'], info['max_demand'], rawdata_train)
    outline =  ["%.2f" % obj_value] + ['%d' % len(loc_chosen)] + ['%.2f' % t] +['%s'%str_loc] + ['%f' % gap]
    outputfile.writelines(','.join(outline) + '\n')
    # print(obj_value, saa)
    outputfile.close()
    print("Run SAA Finished")

def run_SAA_cov(rawdata_train, info,  num_node, num_cov):
    outputfile = open(
        'result/SAA_node%d_cov%d.csv' % ( num_node, num_cov), 'w')
    print(len(rawdata_train))

    outline1 = [ 'obj_value',   'Num_loc', 'sol_time', 'str_loc','gap']
    outputfile.writelines(','.join(outline1) + '\n')
    loc_chosen, obj_value, t, gap = SAA1(rawdata_train, info)
    str_loc = ";".join([str(x) for x in loc_chosen])
    # saa = SAACost(loc_chosen, info['fixed_cost'], info['dist'], info['num_customer'], info['max_demand'], rawdata_train)
    outline =  ["%.2f" % obj_value] + ['%d' % len(loc_chosen)] + ['%.2f' % t] +['%s'%str_loc] + ['%f' % gap]
    outputfile.writelines(','.join(outline) + '\n')
    # print(obj_value, saa)
    outputfile.close()
    print("Run SAA Finished")