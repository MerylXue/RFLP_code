## generate the Figure 5 and Table 1

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df_PUB = pd.read_csv("result/Storm/Kol_cov-Node_49-Cov_2-Beta_0.20.csv",index_col=False)
df_M = pd.read_csv("result/Storm/Moment-Node_49-Cov_2-Beta_0.20.csv",index_col=False)

df =  pd.concat([df_PUB, df_M], ignore_index=True)
print(df.head(10))
### Generate Figure 5
df = df[df['Method'].isin(['Kol_cov','MM_cov','CM_cov'])]
df.rename(columns={'Train_Start_year': 'Instance index'}, inplace=True)
df['Instance index'] = df['Instance index'] - 1996
df = df.replace({'Kol_cov':"PUB-COV",'CM_cov':"CM-COV", "MM_cov":"MM-COV"})
df['Out-of-sample cost (1e6)'] = df['OutofSample'] /1e6

Y = 'Out-of-sample cost (1e6)'

sns.set(style='darkgrid', rc={"axes.facecolor": ".9"}, font="Times New Roman", font_scale=1.2)

seq = ['PUB-COV' , 'MM-COV', 'CM-COV' ]
markers = {'PUB-COV': 's', 'MM-COV': 'X', 'CM-COV': 'd'}
g = sns.relplot(x = 'Instance index', y = 'Out-of-sample cost (1e6)', size = 'Method', size_order = seq,
                color = 'black',  style = 'Method',  markers=markers, data = df)
                    # palette="ch:r=-.5,l=.75")

# g.set_ylim(0.8*min(df[df['Method'].isin(seq)][Y]), 1.5*max(df[df['Method'].isin(seq)][Y]))
# plt.axhline(value, color='black')
# plt.setp(g.get_legend().get_texts(), fontsize='50')
# g.yaxis.get_major_formatter().set_powerlimits((0,1))
# g.legend(loc='center right', bbox_to_anchor=(1, 1.09), ncol=3,
#          fontsize=30, markerscale = 2)  # change the values here to move the legend box

# 设置图例
# plt.legend()

# 设置X轴和Y轴标签
plt.xlabel('Instance Index')
plt.ylabel( 'Out-of-sample cost (1e6)')

plt.savefig("Plot/Figure5.pdf")
# 显示图表
plt.show()


print("----------------------Table 1------------------")

print("Average out-of-sample cost for PUB-COV is %.2f"%(df_PUB['OutofSample'].mean()/1E6))
reliability = 0
for l in range(len(df_PUB)):
    # print(df_PUB['OutofSample'], df_PUB['obj'])
    if df_PUB['OutofSample'].iloc[l] < df_PUB['obj'].iloc[l]:
        reliability += 1
print("Reliability of PUB-COV is %f"%(reliability/df_PUB.shape[0]))


df_MM = df_M[df_M['Method'] == 'MM_cov']

print("Average out-of-sample cost for MM-COV is %.2f"%(df_MM['OutofSample'].mean()/1E6))
reliability = 0
for l in range(len(df_MM)):
    if df_MM['OutofSample'].iloc[l] < df_MM['obj'].iloc[l]:
        reliability += 1
print("Reliability of MM-COV is %f"%(reliability/df_MM.shape[0]))

df_CM = df_M[df_M['Method'] == 'CM_cov']

print("Average out-of-sample cost for CM-COV is %.2f"%(df_CM['OutofSample'].mean()/1E6))


reliability = 0
for l in range(len(df_CM)):
    if df_CM['OutofSample'].iloc[l] < df_CM['obj'].iloc[l]:
        reliability += 1
print("Reliability of CM-COV is %f"%(reliability/df_CM.shape[0]))


min_PUB = 0
min_MM = 0
min_CM = 0
for l in range(len(df_CM)):
    if  df_PUB['OutofSample'].iloc[l] <= min(df_CM['OutofSample'].iloc[l],df_MM['OutofSample'].iloc[l] ):
        min_PUB += 1

    elif df_CM['OutofSample'].iloc[l] <= min( df_PUB['OutofSample'].iloc[l],df_MM['OutofSample'].iloc[l] ):
        min_CM += 1
    else:
        min_MM += 1

print("Frequency of yielding the least-cost RFL design:, PUB-COV: %f, MM-COV: %f, CM-COV: %f"
      %(min_PUB/len(df_PUB),min_MM/len(df_PUB),min_CM/len(df_PUB)))
