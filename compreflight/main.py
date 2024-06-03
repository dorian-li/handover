# !!!模拟出姿态后，此工作被暂停，转而去在实测上做模型去了!!!
# !!!按招标书，后续应还需要根据模拟的姿态+飞机的速度矢量，计算出飞行轨迹，才最终能形成所谓仿真数据集!!!
# 飞行轨迹用于生成飞行过程的背景磁场
# 姿态用于仿真磁干扰场
# 全航态的初衷是FOM的小姿态引起T-L复共线性，因此想仿真包括大姿态、甚至翻转飞行的轨迹
# 下面算法的特色是：生成的姿态变化连续、随机性使得可以模拟多种多样的全航态飞行
# 但它怎么在模型或流程中使用，没有明确的想法，目前是纯讲故事的产物
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from matplotlib import cm
from scipy.ndimage import uniform_filter1d
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm


def make_sprial(start_theta, start_phi, v_angle=0.4, sampling_rate=10):
    # 生成一圈螺旋线
    # start_theta, start_phi: 起始点的球坐标
    # v_angle: 每秒角速度，一般参考实际飞机机动能力的姿态角变化率, deg/s
    # sampling_rate: 采样率, Hz
    end_phi = np.random.default_rng().normal(2 * np.pi, 0.5) + start_phi
    end_theta = np.random.default_rng().normal(0, 0.5) + start_theta
    n_samples = int(
        np.rint((np.rad2deg(end_phi - start_phi) / v_angle) * sampling_rate)
    )

    theta_spiral = np.linspace(start_theta, end_theta, n_samples)
    phi_spiral = np.linspace(start_phi, end_phi, n_samples)

    return theta_spiral, phi_spiral


# 一系列不同缠绕的螺旋线，数量够多时近似覆盖球面，实现全航态
# 球坐标实际上只能建模飞机的朝向（俯仰角Pitch、偏航角Yaw），无法建模飞机的横滚角Roll
# 横滚角采用像弹簧一样的螺旋线模拟
def generate_spirals(n_spirals, init_theta=np.pi / 2, init_phi=0):
    # 此处计算一系列螺旋线的笛卡尔坐标
    # 通过make_sprial生成一圈螺旋线，然后通过rotate_spiral_random旋转螺旋线，使得螺旋线的起始点与上一螺旋线的终点相接
    # 通过smooth_curve对螺旋线进行平滑处理，避免突变
    # n_spirals: 螺旋线数量
    # init_theta, init_phi: 起始点的球坐标
    spirals_xyz = np.array([])
    for _ in tqdm(range(n_spirals)):
        if spirals_xyz.size > 0:
            spirals_xyz = rotate_spiral_random(spirals_xyz)
            prev_end_theta, prev_end_phi = cartesian_to_spherical(*spirals_xyz[-1])

        next_start_theta = init_theta if not spirals_xyz.size else prev_end_theta
        next_start_phi = init_phi if not spirals_xyz.size else prev_end_phi

        theta_spiral, phi_spiral = make_sprial(next_start_theta, next_start_phi)

        spiral_xyz = np.column_stack(spherical_to_cartesian(theta_spiral, phi_spiral))
        spirals_xyz = (
            smooth_curve(spirals_xyz, spiral_xyz) if spirals_xyz.size else spiral_xyz
        )
    print(f"{spirals_xyz.shape=}")

    return spirals_xyz


def smooth_curve(first, next):
    # first, next are all shape of (n, 3), n is the number of samples, 3 is the x,y,z coordinates
    window = 200  # samples
    each_interval = 2000  # samples
    padding = 200  # samples

    connection = len(first)
    curve = np.row_stack((first, next))

    curve[
        connection - each_interval + padding : connection + each_interval - padding
    ] = uniform_filter1d(
        curve[connection - each_interval : connection + each_interval],
        size=window,
        axis=0,
    )[
        padding:-padding
    ]
    # plt.plot(np.row_stack((first, next))[:, 0], label="origin")
    # plt.plot(curve[:, 0], label="smoothed")
    # plt.xlabel("sample [point]")
    # plt.ylabel("x [m]")
    # plt.legend()
    # plt.grid()
    # plt.show()
    return curve


def make_sphere_mesh(r=1):
    # 生成球面网格
    u, v = np.mgrid[0 : 2 * np.pi : 80j, 0 : np.pi : 40j]
    x_sph = r * np.sin(v) * np.cos(u)
    y_sph = r * np.sin(v) * np.sin(u)
    z_sph = r * np.cos(v)

    return x_sph, y_sph, z_sph


def rotate_spiral_random(spiral_cartesian):
    # pitch_roll_random = np.random.default_rng().normal(np.pi / 2, 2, 2).reshape((1, 2))
    # euler_random = np.column_stack((pitch_roll_random, np.array([0])))
    euler_random = np.random.default_rng().normal(0, 0.5, 3).reshape((1, 3))
    r = R.from_euler("xyz", euler_random)
    spiral_cartesian = r.apply(spiral_cartesian)
    return spiral_cartesian


def spherical_to_cartesian(theta, phi, r=1):
    # 球坐标转笛卡尔坐标
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    return x, y, z


def cartesian_to_spherical(x, y, z):
    # 笛卡尔坐标转球坐标
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)

    return theta, phi


def spherical_base_to_ENU(theta, phi):
    return np.pi / 2 - theta, np.pi / 2 - phi


def spherical_to_euler_angle(theta, phi, r=1):
    theta_ENU, phi_ENU = spherical_base_to_ENU(theta, phi)
    # 球坐标转欧拉角
    yaw = np.rad2deg(phi_ENU)
    pitch = np.rad2deg(theta_ENU)
    return yaw, pitch

    # test
    # northward = np.array([0, 1, 0])
    # r = R.from_euler("xyz", [pitch, 0, yaw], degrees=True)

    # direct_euler = r.apply(northward)
    # direct_euler = direct_euler / np.linalg.norm(direct_euler)

    # direct_xyz = spherical_to_cartesian(theta, phi)
    # direct_xyz = direct_xyz / np.linalg.norm(direct_xyz)

    # np.testing.assert_almost_equal(direct_euler, direct_xyz)


def make_roll_around():
    # 横滚角的螺旋线，类似弹簧
    v_angle = 0.8  # deg/s
    sampling_rate = 10  # Hz
    n_samples_around = int(np.rint(360 / v_angle) * sampling_rate)
    roll_rad = np.linspace(-np.pi, np.pi, n_samples_around)
    roll = np.rad2deg(roll_rad)
    return roll


def plot_spiral_with_sphere(sph, sprial):
    # 可视化螺旋线和球面，点太多matplotlib交互很卡，利用web前端可视化
    x_sph, y_sph, z_sph = sph
    x_spiral, y_spiral, z_spiral = sprial
    line_colors = np.linspace(0, 1, len(x_spiral))
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=x_spiral,
                y=y_spiral,
                z=z_spiral,
                mode="lines",
                line=dict(
                    color=line_colors,
                    colorscale="Viridis",
                    cmin=line_colors.min(),
                    cmax=line_colors.max(),
                ),
            )
        ]
    )
    fig.add_surface(
        x=x_sph,
        y=y_sph,
        z=z_sph,
        opacity=0.1,
        surfacecolor=cm.Blues(np.hypot(x_sph, y_sph)),
    )

    fig.update_layout(
        title="3D Colored Line using Plotly",
        scene=dict(xaxis_title="X Axis", yaxis_title="Y Axis", zaxis_title="Z Axis"),
    )
    fig.show()


def preview_drone(roll, pitch, yaw, video=True):
    # 可视化全航态螺旋线下飞机模型的姿态变化
    import pyvista as pv

    def rotation_matrix_to_spatial_transformation_matrix(rotation_matrix):
        """将旋转矩阵转换为空间变换矩阵"""
        if not isinstance(rotation_matrix, np.ndarray):
            rotation_matrix = np.array(rotation_matrix)
        if rotation_matrix.shape != (3, 3):
            raise ValueError(
                f"rotation_matrix shape must be (3, 3): {rotation_matrix.shape}"
            )
        return np.vstack(
            (
                np.hstack((rotation_matrix, np.zeros((3, 1)))),
                np.array([0, 0, 0, 1]),
            )
        )

    def rotate_to_northward(drone):
        init_orientation = np.array([0.0, -142.45333862304688, 0.0])
        to_northward_r = R.align_vectors(
            init_orientation.reshape((1, 3)),
            np.array([(0, 1, 0)]),  # y轴正方向为正北
        )[0]
        to_northward = rotation_matrix_to_spatial_transformation_matrix(
            to_northward_r.as_matrix()
        )
        return drone.transform(to_northward)

    def move_to_center(drone):
        to_center = -1 * np.array(drone.center)
        return drone.translate(to_center)

    def scale_to_actual_size(drone):
        from metalpy.utils.bounds import Bounds

        actual_wingspan = 11  # 米
        actual_length = 8.3  # 米
        actual_height = 2.7  # 米
        bounds = Bounds(drone.bounds)
        actual_size = np.array([actual_wingspan, actual_length, actual_height])
        scale_factor: float = actual_size.max() / bounds.extent.max()
        return drone.scale(scale_factor)

    def get_attitude_rotation(yaw, pitch, roll):
        att_eulers = np.column_stack((pitch, roll, yaw))
        att_eulers[:, 2] = -att_eulers[:, 2]  # ENU
        return R.from_euler(
            "xyz",
            angles=att_eulers,
            degrees=True,
        )

    def preview(drone, yaw, pitch, roll):
        from pathlib import Path
        from uuid import uuid1

        pl = pv.Plotter()
        vehicle = drone.copy()
        pl.add_mesh(vehicle)
        if video:
            cache_file = Path(".video") / f"{uuid1()}.wmv"
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            pl.open_movie(cache_file, quality=10)
        pl.show_grid()
        pl.show_axes()
        pl.add_arrows(
            np.array([0, 0, 3]), np.array([0, 1, 0]), color="lightcoral"
        )  # 指示正北方
        att_rot = get_attitude_rotation(yaw, pitch, roll)
        att_matrixs = att_rot.as_matrix()
        att_matrixs_inv = att_rot.inv().as_matrix()
        for att_matrix, att_matrix_inv in zip(att_matrixs, att_matrixs_inv):
            att_spatial_matrix = rotation_matrix_to_spatial_transformation_matrix(
                att_matrix
            )
            att_spatial_matrix_inv = rotation_matrix_to_spatial_transformation_matrix(
                att_matrix_inv
            )
            vehicle.transform(att_spatial_matrix, inplace=True)
            if video:
                pl.write_frame()
            vehicle.transform(att_spatial_matrix_inv, inplace=True)
            # sleep(0.1)

        pl.close()

    cessna = pv.read(Path(__file__).parent / "cessna_172.stl")
    cessna = rotate_to_northward(cessna)
    cessna = move_to_center(cessna)
    cessna = scale_to_actual_size(cessna)
    preview(cessna, yaw, pitch, roll)


def plot_euler_angles(roll, pitch, yaw):
    plt.plot(yaw, label="Yaw航向角")
    plt.plot(pitch, label="Pitch俯仰角")
    plt.plot(roll, label="Roll横滚角")
    plt.xlabel("采样点 [point]")
    plt.ylabel("姿态角 [deg]")
    plt.legend()
    plt.grid()
    plt.show()


def run():
    spirals_xyz = generate_spirals(50)
    x_spiral, y_spiral, z_spiral = spirals_xyz.T
    sphere_r = 1
    x_sph, y_sph, z_sph = make_sphere_mesh(sphere_r)
    plot_spiral_with_sphere((x_sph, y_sph, z_sph), (x_spiral, y_spiral, z_spiral))
    roll, pitch, yaw = spirals2euler(spirals_xyz)
    plot_euler_angles(roll, pitch, yaw)

    preview_drone(roll, pitch, yaw)


def spirals2euler(spirals_xyz):
    x_spiral, y_spiral, z_spiral = spirals_xyz.T
    theta, phi = cartesian_to_spherical(x_spiral, y_spiral, z_spiral)
    yaw, pitch = spherical_to_euler_angle(theta, phi)
    roll_around = make_roll_around()
    # repeat roll_around until it has the same length as yaw
    roll = np.tile(roll_around, int(np.ceil(len(yaw) / len(roll_around))))[: len(yaw)]
    return roll, pitch, yaw


if __name__ == "__main__":
    run()
