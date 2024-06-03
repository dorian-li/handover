from compreflight import (
    generate_spirals,
    spirals2euler,
    plot_euler_angles,
)

spirals_xyz = generate_spirals(20)
roll, pitch, yaw = spirals2euler(spirals_xyz)
plot_euler_angles(roll, pitch, yaw)
