from __future__ import annotations

import matplotlib.pyplot as plt
from sgl2020 import Sgl2020
from numpy.typing import ArrayLike
from typing_extensions import NamedTuple
import pyvista as pv
from pathlib import Path
from dvmss.data import SensorPos, BackgroundFieldXYZ, PermanentFieldXYZ, InducedFieldXYZ, MagVectorXYZ, LocationWGS84, \
    InertialAttitude, Date
from dvmss.utils.transform import project_vector_3d
from deinterf.utils.data_ioc import DataNDArray

from curr_sim import generate_modified_square_wave
from dvmss.data import (
    DataIoC,  # 依赖容器
    Vehicle,  # 测量平台
)
import numpy as np


def cal_avg_dist(points):
    # 小工具，获取相邻点对之间的平均相对距离
    # points: np.ndarray, shape=(n, 3)
    diffs = np.diff(points, axis=0)  # 计算相邻点之间的差值
    distances = np.sqrt(np.sum(diffs ** 2, axis=1))  # 计算每个差值向量的欧几里得距离

    # 计算平均距离
    average_distance = np.mean(distances)
    return average_distance


class Picker:
    # 小工具，用于输出点击位置，用途：想放置的恒定场源位置、探头位置、载流线缆的过程点等
    def __init__(self, plotter, mesh):
        self.plotter = plotter
        self.mesh = mesh
        self._points = []

    @property
    def points(self):
        """To access all th points when done."""
        return self._points

    def __call__(self, *args):
        picked_pt = np.array(self.plotter.pick_mouse_position())
        direction = picked_pt - self.plotter.camera_position[0]
        direction = direction / np.linalg.norm(direction)
        start = picked_pt - 1000 * direction
        end = picked_pt + 10000 * direction
        point, ix = self.mesh.ray_trace(start, end, first_point=True)
        print(f"{point=}")
        if len(point) > 0:
            self._points.append(point)
            w = self.plotter.add_mesh(pv.Sphere(radius=0.1, center=point), color="red")
        return


# 扩展dvmss，添加载流线缆定义
class Wire(NamedTuple):
    # 载流线缆
    current: ArrayLike  # 线缆电流变化，A，(n,)
    path: ArrayLike  # 线缆积分路径，将线缆划分为n段，各切片的坐标，(n, 3)
    dl: float  # 线缆元素长度，米


# 拓展dvmss，计算线缆电流在探头处产生的磁场三分量
class WireXYZ(DataNDArray):
    # 载流线缆磁干扰场三分量
    @classmethod
    def __build__(cls, container: DataIoC):
        mu_0 = 4 * np.pi * 1e-7  # 真空磁导率，H/m
        r = container[SensorPos]  # (3,)
        r_prime = container[Wire].path  # (n_slice, 3)
        slices_xyz = ((mu_0 / (4 * np.pi)) * container[Wire].dl * (r - r_prime)) / np.abs(
            r - r_prime) ** 3  # (n_slice, 3)
        wire_xyz = np.sum(slices_xyz, axis=0)  # (3,)
        wire_xyz = container[Wire].current[:, None] * wire_xyz[None, :]  # (n_time, 3)
        wire_xyz *= 1e9 # T -> nT
        return cls(wire_xyz)


# 拓展dvmss，替换原有，使得磁场成分包含线缆电流磁场
class MagVectorWithWireXYZ(DataNDArray):
    # 磁三分量探头测量值
    @classmethod
    def __build__(cls, container: DataIoC) -> MagVectorWithWireXYZ:
        bg_xyz = container[BackgroundFieldXYZ]
        perm_xyz = container[PermanentFieldXYZ]
        induced_xyz = container[InducedFieldXYZ]
        wire_xyz = container[WireXYZ]  # 注意id跟随，此处没有处理，如果不把多个线缆合一起表示，那么现有只会加对应id的线缆
        return cls(bg_xyz + perm_xyz + induced_xyz + wire_xyz)


# 拓展dvmss，计算线缆电流磁场的磁总场
class WireTmi(DataNDArray):
    # 磁总场探头线缆电流磁场成分
    @classmethod
    def __build__(cls, container: DataIoC):
        xyz = container[WireXYZ]
        bg_xyz = container[BackgroundFieldXYZ]
        return cls(project_vector_3d(xyz, bg_xyz))


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

# sim curr
wire0_key_points = np.array(
    [
        [0.45526963, 2.3668957, -0.76365083],
        [0.5101477, 0.60580945, -0.6674626],
        [0.42248023, -0.40508768, -0.3082769],
        [0.2477302, -1.2832363, -0.03102143],
        [0.03622945, -2.371268, 0.1544409],
    ]
)
wire0_spline = pv.Spline(wire0_key_points, 1000)
wire0_dl = cal_avg_dist(wire0_spline.points)
_, wire0_curr = generate_modified_square_wave(
    sampling_rate=1,
    high_duration=200,
    low_duration=1000,
    amplitude=3.0,
    duration=len(flt_d),
    noise_level=0.001,
)
wire0_curr += 0.5

data = DataIoC().with_data(
    Vehicle(
        model_3d_path=Path(__file__).parent / "compreflight" / "cessna_172.stl",
        actual_wingspan=11.0,
        actual_length=8.3,
        actual_height=2.7,
        init_heading=[0.0, -142.45333862304688, 0.0],
    ),
    Wire[0](current=wire0_curr, path=wire0_spline.points, dl=wire0_dl),
    InertialAttitude(
        yaw=flt_d["ins_yaw"], pitch=flt_d["ins_pitch"], roll=flt_d["ins_roll"]
    ),
    LocationWGS84(lon=flt_d["lon"], lat=flt_d["lat"], alt=flt_d["utm_z"]),
    Date(year=2015, doy=177),
    SensorPos[0](0, 1, -1),
)
data.add_provider(MagVectorXYZ, MagVectorWithWireXYZ)

plt.figure()
plt.plot(data[WireTmi])
plt.show()
