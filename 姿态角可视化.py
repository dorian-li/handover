import matplotlib.pyplot as plt
from sgl2020 import Sgl2020

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams.update({"font.size": 12})

# 数据准备
surv_d = (
    Sgl2020()
    .line(["1006.06"])
    .source(
        [
            "ins_pitch",
            "ins_yaw",
            "ins_roll",
        ]
    )
    .take()
)
flt_d = surv_d["1006.06"]

plt.plot(flt_d[['ins_pitch', 'ins_yaw', 'ins_roll']], label=['Pitch俯仰角', 'Yaw偏航角', 'Roll横滚角'])
plt.xlabel("采样点 [point]")
plt.ylabel("姿态角 [deg]")
plt.legend()
plt.show()