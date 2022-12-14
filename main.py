from utils import *


data, headers = get_train_data('./data/t3.segd', standard_scale=True)
data_index, data_columns = data.shape[0], data.shape[1]
sampling_rate, nbr_samples = 1 / headers.traces[0].stats.sampling_rate, data.shape[0]
trace_attribute_table = calculate_attributes(data, sampling_rate)
corr = trace_attribute_table.corr()
one_std = calculate_std(trace_attribute_table['polarity'], 1)
two_std = calculate_std(trace_attribute_table['mean_value'], 2)
three_std = calculate_std(trace_attribute_table['mean_value'], 3)
a = trace_attribute_table.iloc[two_std]
b = trace_attribute_table.iloc[one_std]
b_data = data.loc[:, one_std]
plot_trace(b_data)
plt.scatter(trace_attribute_table['mean_value'], trace_attribute_table['main_frequency'])
plt.show()
# 按炮计算属性
# average_energy = (trace_data_table ** 2).sum().sum() / (trace_data_table.shape[0] * trace_data_table.shape[1])
# total_energy = (trace_data_table ** 2).sum().sum()
# index = effective_bandwidth()

trace_attribute_table.to_csv('t3.csv')
