from pathlib import Path
import pyvista as pv

from dvmss.data import (
    DataIoC,  # 依赖容器
    Vehicle,  # 测量平台
    InducedArgs,  # 感应场参数
    MagAgentMesh,  # 体素模型网格
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
    InducedArgs(susceptibility=1.0, cell_size=0.04),
)

pl = pv.Plotter()
pl.add_mesh(data[MagAgentMesh].to_polydata(), opacity=0.7, show_edges=True)
pl.add_axes()
pl.show_grid()
pl.show()
