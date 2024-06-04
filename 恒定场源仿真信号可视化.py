import matplotlib.pyplot as plt
from sgl2020 import Sgl2020


from dvmss.data import (
    DataIoC,  # 依赖容器
    Date,  # 航线日期
    Dipoles,  # 恒定场源参数
    InertialAttitude,  # 飞行姿态
    LocationWGS84,  # 地理位置
    SensorPos,  # 探头位置
    PermanentFieldTmi,  # 磁总场恒定场分量
    PermanentFieldXYZ,  # 磁三分量恒定场分量
)

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams.update({"font.size": 12})

# 以加拿大数据一航线的飞行姿态和轨迹仿真
surv = (
    Sgl2020()
    .line(["1002.02"])
    .source(
        [
            "ins_yaw",
            "ins_pitch",
            "ins_roll",
            "lon",
            "lat",
            "utm_z",
        ]
    )
    .take()
)
flt_d = surv["1002.02"]

# 提供所需的各种信息，内部自动根据依赖关系构建所需项
# 缺少依赖会有异常处理
data = DataIoC().with_data(
    InertialAttitude(
        yaw=flt_d["ins_yaw"], pitch=flt_d["ins_pitch"], roll=flt_d["ins_roll"]
    ),
    LocationWGS84(lon=flt_d["lon"], lat=flt_d["lat"], alt=flt_d["utm_z"]),
    Date(year=2015, doy=177),
    Dipoles(
        {
            "position": [0, 1.9, 0],
            "moment": 154.8,
            "orientation": [0.9848077, 0.0, -0.17364818],
        },
        {
            "position": [0, 2.7, 0],
            "moment": 179,
            "orientation": [-9.84807753e-01, 1.20604166e-16, 1.73648178e-01],
        },
    ),
    SensorPos[0](1, 1, 1),
)

plt.figure(figsize=(9, 6))
plt.subplot(411)
plt.plot(data[PermanentFieldTmi], label="perm tmi")
plt.legend()
plt.grid()
plt.ylabel("磁场强度 [nT]")

plt.subplot(412)
plt.plot(data[PermanentFieldXYZ][:, 0], label="perm vector x")
plt.legend()
plt.grid()
plt.ylabel("磁场强度 [nT]")

plt.subplot(413)
plt.plot(data[PermanentFieldXYZ][:, 1], label="perm vector y")
plt.legend()
plt.grid()
plt.ylabel("磁场强度 [nT]")

plt.subplot(414)
plt.plot(data[PermanentFieldXYZ][:, 2], label="perm vector z")
plt.legend()
plt.grid()
plt.xlabel("采样点 [point]")
plt.ylabel("磁场强度 [nT]")

plt.show()
