from sgl2020 import Sgl2020
from numpy.typing import ArrayLike
from typing_extensions import NamedTuple
import pyvista as pv
from pathlib import Path

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
    distances = np.sqrt(np.sum(diffs**2, axis=1))  # 计算每个差值向量的欧几里得距离

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

wire1_key_points = np.array(
    [
        [-0.38026375, 2.7087429, 0.21086119],
        [-0.5277725, 1.1815339, 0.02534048],
        [-0.27122974, -1.2832363, -0.07505196],
    ]
)
wire1_spline = pv.Spline(wire1_key_points, 1000)
wire1_dl = cal_avg_dist(wire1_spline.points)
t, wire1_curr = generate_modified_square_wave(
    sampling_rate=1,
    high_duration=700,
    low_duration=2000,
    amplitude=1.0,
    duration=len(flt_d),
    noise_level=0.001,
)
wire1_curr += 0.6


data = DataIoC().with_data(
    Vehicle(
        model_3d_path=Path(__file__).parent / "compreflight" / "cessna_172.stl",
        actual_wingspan=11.0,
        actual_length=8.3,
        actual_height=2.7,
        init_heading=[0.0, -142.45333862304688, 0.0],
    ),
    Wire[0](current=wire0_curr, path=wire0_spline.points, dl=wire0_dl),
    Wire[1](current=wire1_curr, path=wire1_spline.points, dl=wire1_dl),
)

pl = pv.Plotter()
pl.add_mesh(data[Vehicle].model_3d, opacity=0.5)
# 小工具，用于输出点击位置，用途：想放置的恒定场源位置、探头位置、载流线缆的过程点等
# picker = Picker(pl, origin)
# pl.track_click_position(picker, side="right")
pl.add_mesh(
    wire0_spline.tube(radius=0.04),
    smooth_shading=True,
    color="orange",
    opacity=0.7,
)
pl.add_mesh(
    wire1_spline.tube(radius=0.04),
    smooth_shading=True,
    color="orange",
    opacity=0.7,
)
pl.show_grid()
pl.show()
