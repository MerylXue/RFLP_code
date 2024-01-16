"""
Code for "Hao Shen, Mengying Xue, and Zuojun Max Shen (2023),
Data-driven Reliable Faciltiy Location Problem, Management Science, Forthcoming",

Last Revised on Jan 12, 2024
@author: Mengying Xue

"""

## this file output the figures in the manuscript

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
def read_data(num_node, num_cov, beta_):
    data_set = [10,25,50,75,100,250,500,750,1000]
    Method_lst = ["Kol","moment"]
    # Method_lst = ["moment","KolNoCov"]

    pd_lst = []
    for num_data in data_set:
        for method in Method_lst:
            if method == 'moment':
                data = pd.read_csv('result/Reliability_%s_Node%d_Data%d_Cov%d.csv' %
                                   (method, num_node, num_data, num_cov),
                                   low_memory=False)
                data = data.rename(columns={'Avg_gap':'Gap'})
                pd_lst.append(data)
            else:
                pd_lst.append(pd.read_csv('result/Reliability_%s_Node%d_Data%d_Cov%d_Beta%.4f.csv' %
                                   (method, num_node, num_data, num_cov, beta_),
                                   low_memory=False))
    data_pd = pd.concat(pd_lst, ignore_index=True)
    data_pd = data_pd.rename(columns={'num_data':"Data size",'out_of_sample_reliability':"Reliability",
                            "in_sample_cost_avg":"In-sample Performance",  "out_of_sample_cost_avg":"Out-of-sample Performance"})

    # data_pd = data_pd.replace({'Kol_cov''CM_cov':"CM", "MM_cov":"MM"})
    # data_pd = data_pd.replace({"Kol_nocov":"Kol", 'CM_nocv':"CM", "MM_nocov":"MM"})
    data_true = pd.read_csv('result/SAA_node%d_cov%d.csv' %
                            (  num_node,num_cov),
                            low_memory=False)
    optimum = data_true['obj_value'].iloc[0]
    data_pd['In-sample Performance'] = data_pd['In-sample Performance']/optimum

    data_pd['Out-of-sample Performance'] = data_pd['Out-of-sample Performance'] / optimum

    return data_pd
#

def plot_fig_Reliability(df, Y,Method_lst,num_node, num_cov, beta_):
    plt.figure(figsize=(13, 10))
    sns.set(style='white', font="Times New Roman", font_scale=3)
    # sns.set_style("white")
    # cmap = sns.cubehelix_palette(rot=-.4)
    g = sns.lineplot(data=df, x="Data size", y=Y, hue='Method', hue_order = Method_lst,
                     style="Method", markers=True, markersize=30)

    plt.axhline(1-beta_, color='black')
    plt.setp(g.get_legend().get_texts(), fontsize='50')
    # g.yaxis.get_major_formatter().set_powerlimits((0,1))
    g.legend(loc='center right', bbox_to_anchor=(1, 1.09), ncol=3,
             fontsize=30, markerscale = 3)  # change the values here to move the legend box

    plt.xscale("log")
    # plt.yscale("log")
    plt.savefig("Plot/%s_Node%d_Cov%d_Beta%.4f.pdf" % (Y, num_node, num_cov, beta_))

    plt.show()


def plot_fig_cost(df, Y, Method_lst, value,num_node, num_cov, beta_):
    plt.figure(figsize=(13, 10))
    sns.set(style='white', font="Times New Roman", font_scale=3)

    g = sns.lineplot(data=df, x="Data size", y=Y, hue='Method',  hue_order =Method_lst,
                     style="Method", markers=True, markersize=30)
    # g.set_ylim(0.8*min(df[df['Method'].isin(['Kol', 'CM', 'MM'])][Y]), 1.5*max(df[df['Method'].isin(['Kol', 'CM', 'MM'])][Y]))
    plt.axhline(value, color='black')
    plt.setp(g.get_legend().get_texts(), fontsize='50')
    # g.yaxis.get_major_formatter().set_powerlimits((0,1))
    g.legend(loc='center right', bbox_to_anchor=(1, 1.09), ncol=3,
             fontsize=30, markerscale = 3)  # change the values here to move the legend box

    plt.xscale("log")
    # plt.yscale("log")
    plt.savefig("Plot/%s_Node%d_Cov%d_Beta%.4f.pdf" % (Y, num_node, num_cov, beta_))

    plt.show()




