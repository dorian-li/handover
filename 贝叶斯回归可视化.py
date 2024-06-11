from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from deinterf.compensator.tmi.linear import Terms, TollesLawson
from deinterf.compensator.tmi.linear.tolles_lawson import ComposableTerm
from deinterf.foundation.sensors import DataNDArray, DirectionalCosine, MagVector, Tmi
from deinterf.metrics.fom import improve_rate, noise_level
from deinterf.utils.data_ioc import DataIoC
from deinterf.utils.filter import fom_bpfilter
from matplotlib.ticker import MaxNLocator
from numpy.typing import ArrayLike
from scipy.signal import butter, filtfilt
from sgl2020 import Sgl2020
from sklearn.linear_model import BayesianRidge

# matplotlib font size
plt.rcParams.update({"font.size": 14})


class Current(DataNDArray):
    def __new__(cls, curr: ArrayLike, Wn: float = None):
        if Wn is not None:
            b, a = butter(6, Wn=Wn, btype="lowpass", output="ba", fs=10)
            curr_filted = filtfilt(b, a, curr)
        else:
            curr_filted = curr
        return super().__new__(cls, curr_filted)


class Wire(ComposableTerm):
    def __build__(self, container: DataIoC) -> np.ndarray:
        curr = container[Current]
        curr_dot = np.gradient(curr)
        dcos = container[DirectionalCosine[0]]
        curr_ret = curr[:, None] * dcos
        curr_dot_ret = curr_dot[:, None] * dcos
        return np.column_stack((curr_ret, curr_dot_ret))


flight = "1006.06"
op = "mag_3_uc"
fl = "d"
surv_d = (
    Sgl2020()
    .line([flight])
    .source(
        [
            op,
            f"flux_{fl}_x",
            f"flux_{fl}_y",
            f"flux_{fl}_z",
            "cur_com_1",
            "cur_ac_hi",
            "cur_ac_lo",
            "cur_tank",
            "cur_flap",
            "cur_strb",
            "cur_srvo_o",
            "cur_srvo_m",
            "cur_srvo_i",
            "cur_heat",
            "cur_acpwr",
            "cur_outpwr",
            "cur_bat_1",
            "cur_bat_2",
            "mag_1_c",
            "ins_yaw",
            "ins_pitch",
            "ins_roll",
            "lon",
            "lat",
        ]
    )
    .take()
)
flt_d = surv_d[flight]

expanded_terms = (
    Terms.Terms_16
    | Wire()[0]
    | Wire()[1]
    | Wire()[2]
    | Wire()[3]
    | Wire()[4]
    | Wire()[5]
    | Wire()[6]
    | Wire()[7]
    | Wire()[8]
    | Wire()[9]
    | Wire()[10]
    | Wire()[11]
    | Wire()[12]
    | Wire()[13]
)

w = np.ones(14) * 0.5
tmi = Tmi(flt_d[op])
fom_data = DataIoC().with_data(
    MagVector(flt_d[f"flux_{fl}_x"], flt_d[f"flux_{fl}_y"], flt_d[f"flux_{fl}_z"]),
    Current[0](flt_d["cur_com_1"], w[0]),
    Current[1](flt_d["cur_ac_hi"], w[1]),
    Current[2](flt_d["cur_ac_lo"], w[2]),
    Current[3](flt_d["cur_tank"], w[3]),
    Current[4](flt_d["cur_flap"], w[4]),
    Current[5](flt_d["cur_strb"], w[5]),
    Current[6](flt_d["cur_srvo_o"], w[6]),
    Current[7](flt_d["cur_srvo_m"], w[7]),
    Current[8](flt_d["cur_srvo_i"], w[8]),
    Current[9](flt_d["cur_heat"], w[9]),
    Current[10](flt_d["cur_acpwr"], w[10]),
    Current[11](flt_d["cur_outpwr"], w[11]),
    Current[12](flt_d["cur_bat_1"], w[12]),
    Current[13](flt_d["cur_bat_2"], w[13]),
)
tl = TollesLawson(terms=expanded_terms, estimator=BayesianRidge())
tmi_clean = tl.fit_transform(fom_data, tmi)
print(f"ir={improve_rate(tmi, tmi_clean, verbose=True)}")

mean_coef = tl.model_.coef_
print(f"{mean_coef=}")
print(f"{mean_coef.shape=}")
stddev_coef = np.sqrt(np.diag(tl.model_.sigma_))
max_range = max(
    abs(mean_coef - 3 * stddev_coef).min(), abs(mean_coef + 3 * stddev_coef).max()
)

# 绘制子图
fig, axes = plt.subplots(nrows=len(mean_coef) // 8 + 1, ncols=8, figsize=(16.6, 12))
axes = axes.flatten()
max_pdf = 0

# 计算所有系数的最大概率密度值
for mean, stddev in zip(mean_coef, stddev_coef):
    x_range = np.linspace(-max_range, max_range, 10000)
    pdf_values = (
        1
        / (stddev * np.sqrt(2 * np.pi))
        * np.exp(-((x_range - mean) ** 2) / (2 * stddev**2))
    )
    max_pdf = max(max_pdf, max(pdf_values))

for i, ax in enumerate(axes):
    if i >= 100:
        break
    x_range = np.linspace(-max_range, max_range, 100000)
    pdf_values = (
        1
        / (stddev_coef[i] * np.sqrt(2 * np.pi))
        * np.exp(-((x_range - mean_coef[i]) ** 2) / (2 * stddev_coef[i] ** 2))
    )
    ax.plot(x_range, pdf_values, label=f"Coefficient {i+1}")
    ax.set_xlim([-max_range, max_range])
    # ax.set_ylim([0, max_pdf * 1.1])
    # 设置科学计数法
    ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    if i < len(axes) - 12:  # 如果不是最后一个子图
        ax.set_xticklabels([])  # 隐藏 x 轴的标签

plt.tight_layout(w_pad=0, h_pad=-1.5)
plt.show()