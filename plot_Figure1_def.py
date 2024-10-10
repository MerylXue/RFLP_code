
from Plot.plot import read_data_NoCov, plot_fig_Reliability_NoCov, plot_fig_cost_NoCov
data_set =  [10,25,50,75,100,250,500, 1000]

num_node = 10
mu = 0.4
truncate = True
beta_ = 0.1

df = read_data_NoCov(num_node, 1, beta_, mu, truncate,  ["Wasserstein", "KolNoCov"], data_set)
Y = 'Reliability'
plot_fig_Reliability_NoCov(df, Y, ["PUB","Wass"],num_node, beta_, mu, truncate)

Y = 'In-sample Performance'
plot_fig_cost_NoCov(df, Y, ["PUB","Wass"], 1, num_node, beta_, mu, truncate)

Y = 'Out-of-sample Performance'
plot_fig_cost_NoCov(df, Y, ["PUB","Wass"], 1, num_node,  beta_, mu, truncate)
