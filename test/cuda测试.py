# test/03-cuda测试.py
import sys
print(f"🔍 当前Python解释器路径: {sys.executable}")
try:
    import torch
    print(f"✅ PyTorch 加载成功！版本：{torch.__version__}")
    print(f"✅ CUDA 状态：{torch.cuda.is_available()}（CPU版显示False正常）")
    print(f"✅ CUDA 设备数：{torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"✅ CUDA 设备名称：{torch.cuda.get_device_name(0)}")
    else:
        print("⚠️  CUDA不可用，将使用CPU")
except Exception as e:
    print(f"❌ PyTorch 加载失败：{e}")

