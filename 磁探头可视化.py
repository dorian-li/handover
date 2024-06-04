from pathlib import Path
import pyvista as pv

from dvmss.data import (
    DataIoC,  # 依赖容器
    Dipoles,  # 恒定场源参数
    Vehicle,  # 测量平台
    SensorPos,  # 探头位置
)

# 提供所需的各种信息，内部自动根据依赖关系构建所需项
# 缺少依赖会有异常处理
data = DataIoC().with_data(
    Vehicle(
        model_3d_path=Path(__file__).parent / "compreflight" / "cessna_172.stl",
        actual_wingspan=11.0,
        actual_length=8.3,
        actual_height=2.7,
        init_heading=[0.0, -142.45333862304688, 0.0],
    ),
    SensorPos[0](0, 1, -1),
)

pl = pv.Plotter()
pl.add_mesh(data[Vehicle].model_3d, opacity=0.5)
pl.add_mesh(
    pv.Arrow(start=data[SensorPos], direction=(0, 1, 0), shaft_radius=0.1),
    color="tab:red",
    opacity=0.7,
)
pl.add_axes()
pl.show_grid()
pl.show()
