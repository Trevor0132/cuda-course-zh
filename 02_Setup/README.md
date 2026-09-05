# 环境安装指南

1. 首先执行更新命令：`sudo apt update && sudo apt upgrade -y && sudo apt autoremove`，然后前往 [CUDA 下载页面](https://developer.nvidia.com/cuda-downloads)。
2. 根据你所使用的设备配置选择对应的选项：
   - 操作系统 (Operating System)
   - 架构 (Architecture)
   - 发行版 (Distribution)
   - 版本 (Version)
   - 安装类型 (Installer Type)
3. 在 “runfile” 部分，你需要运行类似如下的命令：

```bash
wget https://developer.download.nvidia.com/compute/cuda/12.6.0/local_installers/cuda_12.6.0_560.28.03_linux.run
sudo sh cuda_12.6.0_560.28.03_linux.run
```

4. 安装完成后，运行 `nvcc --version` 应能正确显示 NVIDIA CUDA 编译器的版本信息。
   同时运行 `nvidia-smi` 确保驱动能够正确识别显卡设备与最高支持的 CUDA 版本。

> **注意：** `nvidia-smi` 中显示的 CUDA 版本是指当前 **显卡驱动** 所能支持的最高 CUDA 版本，而非实际安装的 Toolkit 版本。实际安装的 CUDA Toolkit 版本由 `nvcc --version` 显示。两者可能存在差异。如果 `nvidia-smi` 显示的版本低于 `nvcc` 的版本，可能会因为驱动版本过低而导致运行时错误。

5. 如果运行 `nvcc` 提示未找到命令，先执行 `echo $SHELL`。如果是 `/bin/bash`，请将以下环境变量写入 `~/.bashrc`；如果是 `/bin/zsh`，请写入 `~/.zshrc`：

```bash
export PATH=/usr/local/cuda-12.6/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda/lib64
```

保存后执行 `source ~/.zshrc` 或 `source ~/.bashrc` 刷新环境，再次尝试 `nvcc -V`。

## 备用方案：一键安装脚本

- 直接执行本目录下的 Shell 脚本：`./cuda-installer.sh`

## WSL2 配置指南

- 参考外部技术文档：[在 WSL2 上完整安装 Ubuntu、CUDA、cuDNN 及 PyTorch 指南](https://medium.com/@omkarpast/technical-documentation-for-clean-installation-of-ubuntu-cuda-cudnn-and-pytorch-on-wsl2-9b265a4b8821)
