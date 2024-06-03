import contextily as ctx
import geopandas as gpd
import matplotlib.pyplot as plt
from sgl2020 import Sgl2020

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams.update({"font.size": 12})

# 数据准备
surv_d = (
    Sgl2020()
    .line(["1006.06"])
    .source(
        [
            "lon",
            "lat",
            "utm_z",
        ]
    )
    .take()
)
flt_d = surv_d["1006.06"]

# 2D可视化，带卫星底图
# 需要联网，容易g
# Convert the DataFrame to a GeoDataFrame with longitude and latitude
gdf = gpd.GeoDataFrame(
    flt_d,
    geometry=gpd.points_from_xy(flt_d.lon, flt_d.lat),
    # has_electro,
    # geometry=gpd.points_from_xy(has_electro.lon, has_electro.lat),
)

# Set the coordinate reference system (CRS) to WGS84 for latitude and longitude
gdf.crs = "epsg:4326"
# Plotting
fig, ax = plt.subplots()
gdf.plot(ax=ax, color="tab:red")

# Add a basemap with satellite imagery
ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery, crs=gdf.crs)
plt.xlabel("经度位置 [deg]")
plt.ylabel("纬度位置 [deg]")

plt.show()

# 3D轨迹
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot(flt_d["lon"], flt_d["lat"], flt_d["utm_z"], lw=1)
ax.set_zlim(bottom=flt_d["utm_z"].min() - 500)
ax.set_xlabel("经度位置 [deg]")
ax.set_ylabel("纬度位置 [deg]")
ax.set_zlabel("飞行高度 [m]")
# make the gap between axis number and label
ax.xaxis.labelpad = 7
ax.yaxis.labelpad = 7
ax.zaxis.labelpad = 6
# 让x axis number稀疏一点
ax.locator_params(axis="x", nbins=4)
ax.locator_params(axis="y", nbins=4)
plt.show()