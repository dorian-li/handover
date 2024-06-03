from compreflight import generate_spirals, make_sphere_mesh, plot_spiral_with_sphere

spirals_xyz = generate_spirals(10)
x_spiral, y_spiral, z_spiral = spirals_xyz.T
sphere_r = 1
x_sph, y_sph, z_sph = make_sphere_mesh(sphere_r)
plot_spiral_with_sphere((x_sph, y_sph, z_sph), (x_spiral, y_spiral, z_spiral))
