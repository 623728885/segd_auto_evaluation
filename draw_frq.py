import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pylab import mpl

mpl.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
mpl.rcParams['axes.unicode_minus'] = False  # 显示负号

Fs = 1000  # 采样频率
Ts = 1 / Fs  # 采样区间
x = np.arange(0, 1, Ts)  # 时间向量，1400个

y = 5 * np.sin(2 * np.pi * 450 * x) + np.cos(2 * np.pi * 300 * x) + 4 * np.sin(2 * np.pi * 110 * x)

N = 1000
frq = np.fft.rfftfreq(N, Ts)

fft_y = np.fft.rfft(y)

abs_y = np.abs(fft_y)  # 取复数的绝对值，即复数的模(双边频谱)
db_y = 20 * np.log10(abs_y)
angle_y = np.angle(fft_y, deg=True)  # 取复数的弧度,并换算成角度
gui_y = abs_y / N  # 归一化处理（双边频谱）
gui_half_y = gui_y[range(int(N / 2))]  # 由于对称性，只取一半区间（单边频谱）

# 画出原始波形的前50个点
plt.subplot(231)
plt.plot(frq[0:50], y[0:50])
plt.title('原始波形')

# 画出双边未求绝对值的振幅谱
plt.subplot(232)
plt.plot(frq, fft_y, 'black')
plt.title('双边振幅谱(未求振幅绝对值)', fontsize=8, color='black')

# 画出双边求绝对值的振幅谱
plt.subplot(233)
plt.plot(frq, abs_y, 'r')
plt.title('双边振幅谱(未归一化)', fontsize=8, color='red')

# 画出双边相位谱
plt.subplot(234)
plt.plot(frq[0:50], angle_y[0:50], 'violet')
plt.title('双边相位谱(未归一化)', fontsize=8, color='violet')

# 画出双边振幅谱(归一化)
plt.subplot(235)
plt.plot(frq, db_y, 'g')
plt.title('双边振幅谱(归一化)', fontsize=8, color='green')

plt.show()
