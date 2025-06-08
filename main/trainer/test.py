from pathlib import Path

# 获取当前脚本文件所在的目录
current_dir = Path(__file__).resolve().parent
print(current_dir)