import matplotlib.pyplot
import pandas as pd
from segd_read_core import _read_segd
import numpy as np
from scipy import fft, interpolate
from sklearn import preprocessing
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt


# def minmax_scale2d(data):
#     maxvalue = np.max(data)
#     minvalue = np.min(data)
#     cache = 2 * (data - minvalue) / (maxvalue - minvalue) - 1
#     return cache


def get_train_data(file_name, cut_assist_trace=True, standard_scale=False):
    """
    该函数用于去除辅助道，并提供对数据的标准化处理
    :param file_name: segd数据文件名
    :param cut_assist_trace: 是否切除辅助道
    :param standard_scale: 是否对数据进行标准化处理
    :return: 返回处理后的数据
    """
    data_in = _read_segd(file_name, merge=True)
    trace_head_in = _read_segd(file_name, headonly=True, merge=False)
    trace_head_table = pd.DataFrame(columns=range(0, len(trace_head_in.traces) - 1))
    nbr_samples, nbr_traces = trace_head_in.traces[0].stats.npts, len(trace_head_in.traces)
    if cut_assist_trace:
        trace_sample_start = nbr_samples * 3
        trace_start = 3
        nbr_traces -= 3
    else:
        trace_sample_start, trace_start = 0, 0

    # for i in range(trace_start, len(trace_head_in.traces)):
    #     for key in trace_head_index:
    #         head = trace_head_in.traces[i].stats[key]
    #         trace_head_table.columns[i] = pd.Series(head, index=trace_head_index)

    data_out = data_in.traces[0].data[trace_sample_start:].reshape((nbr_samples, nbr_traces), order='F')
    if standard_scale:
        scaler = preprocessing.StandardScaler().fit(data_out)
        data_out = scaler.transform(data_out)
    return data_out, trace_head_in


def calculate_attributes(index, columns):
    df = pd.DataFrame(data, index=range(index), columns=range(columns))
    mean_value = df.abs().mean().rename('mean_value')
    main_frequency = pd.Series(index=range(data.shape[1]), name='main_frequency', dtype='float64')
    average_energy = pd.Series(index=range(data.shape[1]), name='average_energy', dtype='float64')
    total_energy = pd.Series(index=range(data.shape[1]), name='total_energy', dtype='float64')
    for col in df.columns:
        fx = fft.rfftfreq(nbr_samples, sampling_rate)
        fy = np.abs(fft.rfft(df.iloc[:, col].to_numpy()))
        main_frequency[col] = fx[np.where(fy == np.max(fy))[0][0]]
        average_energy[col] = (df.iloc[:, col] ** 2).sum() / df.shape[0]
        total_energy[col] = (df.iloc[:, col] ** 2).sum()
    attribute_df = pd.concat([mean_value, main_frequency, average_energy, total_energy], axis=1)
    return attribute_df


def effective_bandwidth(index, columns):
    df = pd.DataFrame(data, index=range(index), columns=range(columns))
    freq_range = fft.rfftfreq(nbr_samples, sampling_rate)
    amplitude = np.abs(fft.rfft(df.mean(axis=1).to_numpy()))
    db = 20 * np.log10(amplitude)
    main_frequency = freq_range[np.where(db == np.max(db))[0][0]]
    max_amp = np.max(db)
    interpolate_func = interpolate.interp1d(freq_range, db, 'cubic')
    poly_func = np.poly1d(np.polyfit(freq_range, db, 8))
    range_x = range(1, 500)
    interpolate_y = interpolate_func(range_x)
    index = []
    for i in range(0, 499):
        if interpolate_y[i] > (max_amp * 0.707):
            index.append(i)
    return index

    # plt.plot(range_x, interpolate_db_y)
    # plt.show()


data, headers = get_train_data('./data/t3.segd', standard_scale=False)
data_index, data_columns = data.shape[0], data.shape[1]

sampling_rate, nbr_samples = 1 / headers.traces[0].stats.sampling_rate, data.shape[0]
trace_attribute_table = calculate_attributes(data_index, data_columns)
std = trace_attribute_table['mean_value'].std()
mean = trace_attribute_table['mean_value'].mean()
one_std = (abs(trace_attribute_table['mean_value'] - mean) - 1 * std) > 0
one_std = one_std[one_std]
two_std = (abs(trace_attribute_table['mean_value'] - mean) - 2 * std) > 0
two_std = two_std[two_std]
three_std = (abs(trace_attribute_table['mean_value'] - mean) - 3 * std) > 0
three_std = three_std[three_std]
a = trace_attribute_table.iloc[three_std.index]
three_alpha = trace_attribute_table[[abs(trace_attribute_table['mean_value'] - mean) > 3 * std]]

# 按炮计算属性
# average_energy = (trace_data_table ** 2).sum().sum() / (trace_data_table.shape[0] * trace_data_table.shape[1])
# total_energy = (trace_data_table ** 2).sum().sum()
# index = effective_bandwidth()

trace_attribute_table.to_csv('t3.csv')
