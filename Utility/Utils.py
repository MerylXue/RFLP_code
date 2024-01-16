"""
Code for "Hao Shen, Mengying Xue, and Zuojun Max Shen (2023),
Data-driven Reliable Faciltiy Location Problem, Management Science, Forthcoming",

Last Revised on Jan 12, 2024
@author: Mengying Xue

"""


import numpy as np


# sort the distance from customer i to location subset sigma in ascending order
def DistSort(i,sigma, d):
    temp = [(j, d[i][j]) for j in sigma]
    temp_sort = sorted(temp, key=lambda a:a[1])
    return [a[0] for a in temp_sort]

# sort the distance from customer i to location subset which operates normally
#  in ascending order
def DistSortDisrupt(i,sigma, d, xi):
    xi = list(xi) + [1]
    d_disrupt = [d[i][j] if xi[j] > 0 else -1 for j in sigma]
    min_d = 1000000
    idx = 0
    for i in range(len(d_disrupt)):
        if d_disrupt[i] >= 0 and d_disrupt[i] < min_d:
            min_d = d_disrupt[i]
            idx = i

    return sigma[idx], min_d

# sort the demand samples for customer i in k with 0, max_demand_i, in ascending orders
def DemandSort(max_demand_i, mu_i_k):
    if 0 not in mu_i_k:
        temp = [(0, 0), (len(mu_i_k)+1, max_demand_i)] + [(n+1, mu_i_k[n]) for n in range(len(mu_i_k))]
    elif max_demand_i not in mu_i_k:
        temp = [(len(mu_i_k) + 1, max_demand_i)] + [(n , mu_i_k[n]) for n in range(len(mu_i_k))]
    else:
        temp =   [(n , mu_i_k[n]) for n in range(len(mu_i_k))]
    temp_sort = sorted(temp, key=lambda a: a[1])
    return [a[1] for a in temp_sort]




## calculate the marginal probabilities for each covariate
def diff_marginal_prob_cov(cdf_emp_prob_cov, k, eps):
    return min(1,cdf_emp_prob_cov[k]-cdf_emp_prob_cov[k-1]+eps)



# bootstrap sampling data from given data set as the training data,
# let the remaining data as the test data
def BootstrapSample(raw_data):
    num_data = len(raw_data)
    num_length = 0
    index_train = []
    index_test  =[]
    while num_length <= 0:
        index_train = np.random.choice(np.array(range(num_data)), size = num_data)
        index_test = list(set(range(num_data)).difference(set(index_train)))
        num_length = len(index_test)

    train_data = raw_data.iloc[list(sorted(index_train))]
    test_data = raw_data.iloc[list(sorted(index_test))]
    return train_data, test_data


# ## K-fold cross validation sampling
# def CrossValidateSample(raw_data, K):
#     num_data = len(raw_data)
#     num_test = int(num_data/K)
#     train_data = []
#     test_data = []
#     for k in range(K):
#         test_data.append(raw_data[k*num_test:(k+1)*num_test])
#         train_data.append(raw_data[0:k*num_test]+raw_data[(k+1)*num_test:])
#
#     return train_data, test_data
#
#
# def TimeSeriesSample(raw_data, t_length):
#     num_data = len(raw_data)
#     #randomly select a period with t_length
#     start_t = np.random.choice(np.array(range(num_data - t_length)))
#
#     train_data = raw_data[start_t: int(start_t + t_length/2)]
#     test_data = raw_data[int(start_t + t_length/2): int(start_t + t_length)]
#
#     return train_data, test_data

# return the value for f(x) = 1 if x<threshold value, else,  f(x)=0
def positive_num(x, threshold):
    if x < threshold:
        return 1
    else:
        return 0


# return the true/false value of whether s is a number
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False

#function for evaluating damage in the weather data
# turn the value of loss into numbers
def ConvertNumber(obj):

    if type(obj) == float:
        return 0
    else:
        number = 0
        damage= str(obj)
        unit = damage[-1]
        if is_number(unit):
            return float(damage)
        else:
            try:
                if(len(damage) > 1):
                    number = float(damage[0:len(damage)-1])
                else:
                    number = 1
            except:
                print(obj, damage)

            if unit == 'K' or unit == 'k':
                return number * 1000
            elif unit == 'M' or unit == 'm':
                return number * 1000000
            elif unit == 'B' or unit == 'b':
                return number * 1000000000