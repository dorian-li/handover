from compreflight import preview_drone
from sgl2020 import Sgl2020
import os

# winget install ffmpeg就可以安装
os.environ["IMAGEIO_FFMPEG_EXE"] = (
    r"C:\Users\Dorian\AppData\Local\Microsoft\WinGet\Links\ffmpeg.exe"  # ffmpeg路径用于保存视频
)

surv_d = Sgl2020().line(["1006.06"]).source(["ins_pitch", "ins_yaw", "ins_roll"]).take()
flt_d = surv_d["1006.06"]
preview_drone(
    roll=flt_d["ins_roll"], pitch=flt_d["ins_pitch"], yaw=flt_d["ins_yaw"], video=True
)  # video=True将录制视频并保存在.video文件夹中
