from pathlib import Path
import matplotlib.pyplot as plt
from sgl2020 import Sgl2020


from dvmss.data import (
    DataIoC,  # 依赖容器
    Date,  # 航线日期
    InducedArgs,  # 感应场源参数
    InertialAttitude,  # 飞行姿态
    LocationWGS84,  # 地理位置
    SensorPos,  # 探头位置
    Vehicle,  # 测量平台
    InducedFieldTmi,  # 磁总场感应场分量
    InducedFieldXYZ,  # 磁三分量感应场分量
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
    Vehicle(
        model_3d_path=Path(__file__).parent / "compreflight" / "cessna_172.stl",
        actual_wingspan=11.0,
        actual_length=8.3,
        actual_height=2.7,
        init_heading=[0.0, -142.45333862304688, 0.0],
    ),
    SensorPos[0](1, 1, 1),
    InducedArgs(susceptibility=1.0, cell_size=0.04),
)

plt.figure(figsize=(9, 6))
plt.subplot(411)
plt.plot(data[InducedFieldTmi], label="induced tmi")
plt.legend()
plt.grid()
plt.ylabel("磁场强度 [nT]")

plt.subplot(412)
plt.plot(data[InducedFieldXYZ][:, 0], label="induced vector x")
plt.legend()
plt.grid()
plt.ylabel("磁场强度 [nT]")

plt.subplot(413)
plt.plot(data[InducedFieldXYZ][:, 1], label="induced vector y")
plt.legend()
plt.grid()
plt.ylabel("磁场强度 [nT]")

plt.subplot(414)
plt.plot(data[InducedFieldXYZ][:, 2], label="induced vector z")
plt.legend()
plt.grid()
plt.xlabel("采样点 [point]")
plt.ylabel("磁场强度 [nT]")

plt.show()
