
from Plot.plot import read_data_NoCov, plot_fig_Reliability_NoCov, plot_fig_cost_NoCov
data_set =[10,25,50,75,100,250,500,750,1000]
num_node = 10
mu = 1.6
truncate = True
beta_ = 0.1

Method_lst = ["KolNoCov","moment"]
df = read_data_NoCov(num_node, 1, beta_, mu, truncate,  Method_lst, data_set)
Y = 'Reliability'
plot_fig_Reliability_NoCov(df, Y, ["PUB","MM","CM"],num_node, beta_, mu, truncate)

Y = 'In-sample Performance'
plot_fig_cost_NoCov(df, Y, ["PUB","MM","CM"], 1, num_node, beta_, mu, truncate)

Y = 'Out-of-sample Performance'
plot_fig_cost_NoCov(df, Y, ["PUB","MM","CM"], 1, num_node,  beta_, mu, truncate)
