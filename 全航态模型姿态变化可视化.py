from compreflight import (
    generate_spirals,
    spirals2euler,
    preview_drone,
)
import os

# winget install ffmpeg就可以安装
os.environ["IMAGEIO_FFMPEG_EXE"] = (
    r"C:\Users\Dorian\AppData\Local\Microsoft\WinGet\Links\ffmpeg.exe"  # 提供ffmpeg用于保存视频
)
spirals_xyz = generate_spirals(20)
roll, pitch, yaw = spirals2euler(spirals_xyz)
preview_drone(roll, pitch, yaw, video=False)
