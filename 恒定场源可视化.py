from pathlib import Path
import pyvista as pv

from dvmss.data import (
    DataIoC,  # 依赖容器
    Dipoles,  # 恒定场源参数
    Vehicle,  # 测量平台
)

# 提供所需的各种信息，内部自动根据依赖关系构建所需项
# 缺少依赖会有异常处理
data = DataIoC().with_data(
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
    Vehicle(
        model_3d_path=Path(__file__).parent / "compreflight" / "cessna_172.stl",
        actual_wingspan=11.0,
        actual_length=8.3,
        actual_height=2.7,
        init_heading=[0.0, -142.45333862304688, 0.0],
    ),
)

pl = pv.Plotter()
pl.add_mesh(data[Vehicle].model_3d, opacity=0.5)
perm_sources_pos = data[Dipoles].pos  # (n_sources, 3)
for pos in perm_sources_pos:
    pl.add_mesh(pv.Sphere(radius=0.2).translate(pos), color="red")
pl.add_axes()
pl.show_grid()
pl.show()
