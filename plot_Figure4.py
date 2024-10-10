
from Plot.plot import read_data, plot_fig_Reliability, plot_fig_cost
data_set =[10,25,50,75,100,250,500,750,1000]
num_node = 10
num_cov = 2
beta_ = 0.1

Method_lst = ["Kol","moment"]
df = read_data(num_node, num_cov, beta_ ,  Method_lst, data_set)
Y = 'Reliability'
plot_fig_Reliability(df, Y, ["PUB-COV","MM-COV","CM-COV"],num_node, num_cov, beta_)

Y = 'In-sample Performance'
plot_fig_cost(df, Y, ["PUB-COV","MM-COV","CM-COV"],  1, num_node,  num_cov,beta_)

Y = 'Out-of-sample Performance'
plot_fig_cost(df, Y, ["PUB-COV","MM-COV","CM-COV"],  1, num_node,  num_cov, beta_)
