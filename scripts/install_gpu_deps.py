import subprocess
import sys

def run_command(cmd):
    """执行命令并实时输出"""
    print(f"执行: {' '.join(cmd)}")
    result = subprocess.run(cmd, text=True, capture_output=True)
    print(result.stdout)
    if result.stderr:
        print("错误输出:", result.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"命令执行失败: {' '.join(cmd)}")
    return result

def main():
    print("开始安装支持 RTX 4060 的 GPU 依赖...")
    
    # 确保 uv 可用
    try:
        run_command(["uv", "--version"])
    except FileNotFoundError:
        print("错误: 未找到 uv 工具，请先安装 uv (pip install uv or install via official installer).")
        sys.exit(1)
    
    # 步骤1: 移除已存在的 flagembedding（避免自动安装CPU版torch）
    print("步骤1: 从 pyproject.toml 中移除 flagembedding...")
    run_command(["uv", "remove", "flagembedding"])
    
    # 步骤2: 更新 pyproject.toml 添加 GPU 版本 torch 依赖和源配置
    print("步骤2: 配置 pyproject.toml 使用 CUDA 12.8 源...")
    pyproject_content = '''
[project]
name = "dataset-rag"
version = "0.1.0"
description = "Add your description here"
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.135.1",
    "flagembedding>=v1.3.5",
    "grandalf>=0.8",
    "langchain>=1.2.13",
    "langchain-community>=0.4.1",
    "langchain-openai>=1.1.11",
    "langgraph>=1.1.3",
    "loguru>=0.7.3",
    "magic-pdf>=1.3.12",
    "minio>=7.2.20",
    "modelscope>=1.35.1",
    "numpy>=2.4.3",
    "pandas>=3.0.1",
    "pymilvus>=2.6.10",
    "pymilvus-model>=0.3.2",
    "pymongo>=4.16.0",
    "python-dotenv>=1.2.2",
    "python-multipart>=0.0.22",
    "regex>=2026.2.28",
    "torch>=2.10.0",
    "torchvision>=0.25.0",
    "torchaudio>=2.10.0",
    "uvicorn>=0.42.0",
]

[tool.uv.sources]
# 强制从 NVIDIA 源安装
torch = { index = "pytorch-cuda" }
torchvision = { index = "pytorch-cuda" }
torchaudio = { index = "pytorch-cuda" }

[[tool.uv.index]]
name = "pytorch-cuda"
url = "https://download.pytorch.org/whl/cu128"
explicit = true
'''
    
    with open("pyproject.toml", "w", encoding="utf-8") as f:
        f.write(pyproject_content.strip())
    print("已更新 pyproject.toml 配置。")
    
    # 步骤3: 删除锁文件并重新生成
    print("步骤3: 删除 uv.lock 并重新锁定依赖...")
    try:
        import os
        os.remove("uv.lock")
        print("已删除 uv.lock")
    except FileNotFoundError:
        pass
    
    run_command(["uv", "lock"])
    
    # 步骤4: 同步安装所有依赖
    print("步骤4: 同步并重新安装依赖...")
    run_command(["uv", "sync", "--reinstall"])
    
    # 步骤5: 验证 GPU 可用性
    print("步骤5: 验证 PyTorch GPU 是否可用...")
    run_command(["uv", "run", "python", "-c", "import torch; print('GPU:', torch.cuda.is_available())"])
    
    print("\n✅ 所有操作已完成！GPU 环境已就绪。")

if __name__ == "__main__":
    main()
