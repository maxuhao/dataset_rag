# GPU 依赖安装脚本说明

本脚本用于在配备 NVIDIA 显卡（如 RTX 4060）的 Windows 环境中，自动配置并安装支持 GPU 的 PyTorch 及相关依赖。

## 脚本功能
- 自动移除可能导致 CPU 版本 torch 被安装的冲突项
- 配置 `pyproject.toml` 以强制从 NVIDIA 官方源下载 CUDA 12.8 版本的 `torch`, `torchvision`, `torchaudio`
- 删除旧锁文件并重新生成依赖关系
- 同步安装所有依赖包
- 最终验证 GPU 是否可用

## 兼容性要求
| 组件 | 要求 |
|------|------|
| **显卡** | NVIDIA GeForce RTX 4060 或其他 Ada Lovelace 架构显卡 |
| **CUDA 支持** | 驱动需支持 CUDA 11.8+，推荐使用 12.1+ |
| **Python 版本** | 3.11 或更高版本（建议 3.11） |
| **包管理器** | [`uv`](https://docs.astral.sh/uv/) 已安装 |

> ✅ 当前配置 `cu128 + Python 3.11` 是 RTX 4060 的最佳组合，无需修改。

## 使用方法
1. 确保已安装 `uv`：
   ```bash
   pip install uv
   # 或通过官方方式安装
   ```
2. 运行脚本：
   ```bash
   python scripts/install_gpu_deps.py
   ```
3. 等待执行完成，最终输出应显示：
   ```
   GPU: True
   ✅ 所有操作已完成！GPU 环境已就绪。
   ```

## 注意事项
- 若中途失败，请检查网络连接或 NVIDIA 源是否可访问
- 确保系统已正确安装最新版 NVIDIA 驱动程序
- 不要手动中断依赖安装过程
