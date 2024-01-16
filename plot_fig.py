from Plot.plot import read_data, plot_fig_Reliability, plot_fig_cost


num_node = 10
beta_=0.1

# mu= 1.6
# truncate = 1
num_cov = 2

# df = read_raw_data(num_node, num_cov, beta_, mu, truncate)

# print(df)
Method_lst = ['Kol_cov','MM_cov','CM_cov']
# Method_lst = ['Kol','Wass']


df = read_data(num_node, num_cov, beta_)
df = df[df['Method'].isin(Method_lst)]
# print(df)
Y = 'Reliability'
plot_fig_Reliability(df, Y, Method_lst,num_node, num_cov, beta_)

Y = 'In-sample Performance'
plot_fig_cost(df, Y, Method_lst, 1, num_node, num_cov, beta_)

Y = 'Out-of-sample Performance'
plot_fig_cost(df, Y, Method_lst, 1, num_node, num_cov, beta_)


# plot_fig_cost_box(df, Y, num_node, num_cov, beta_, mu, truncate)