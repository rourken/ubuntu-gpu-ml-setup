# Ubuntu GPU Machine Learning Setup

This repository provides a step-by-step guide to setting up a development environment for computer vision and large language models (LLMs) on **Ubuntu 24.04.3 LTS** with **NVIDIA GPU** support.

It includes the installation of NVIDIA drivers, CUDA 12.6, OpenBLAS, cuBLAS, NumPy, SciPy, FilterPy, OpenCV, FFmpeg, PyTorch, llama-cpp-python, and additional libraries in a Python virtual environment.

---

## ✨ Features

- Installs **CUDA 12.6** and NVIDIA drivers for GPU acceleration.
- Configures **OpenBLAS** and **cuBLAS** for optimized numerical computations.
- Builds **OpenCV 4.12.0** and **FFmpeg 7.1.2** with CUDA support.
- Installs **PyTorch** and **llama-cpp-python** with GPU acceleration.
- Uses **uv** for lightning-fast Python package installation and virtual environment management (instead of standard pip).

---

## 🖥️ Prerequisites

- **OS**: Ubuntu 24.04.3 LTS  
- **GPU**: NVIDIA GPU (e.g., RTX 3000 with CUDA Compute Capability 8.6)  
- **Disk Space**: At least 50 GB free  
- **RAM**: 16 GB or more recommended  
- **Internet**: Required for downloading packages  
- **Permissions**: `sudo` access required  

---

## 🚀 Installation

Follow the detailed instructions in [INSTALL.md](INSTALL.md):

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/ubuntu-gpu-ml-setup.git
   cd ubuntu-gpu-ml-setup
   ```
2. Follow each step in INSTALL.md, executing the commands in your terminal.

3. Reboot the system after installing the NVIDIA drivers (Step 5).

4. After installation, activate the virtual environment:

```bash
source ~/venv/bin/activate
```

---

## ✅ Verification
Run the following commands to verify your setup:
```bash
# Check NVIDIA drivers
nvidia-smi

# Check CUDA installation
nvcc --version

# Check uv installation (Package Manager)
uv --version

# Verify NumPy
python3 -c "import numpy as np; np.show_config()"

# Verify SciPy
python3 -c "import scipy; scipy.show_config()"

# Verify FilterPy
python3 -c "import numpy as np; from filterpy.kalman import KalmanFilter; print(KalmanFilter(dim_x=2, dim_z=1))"

# Verify OpenCV
python3 -c "import cv2; print(cv2.getBuildInformation())"

# Verify PyTorch
python3 -c "import torch; print(torch.cuda.is_available())"

# Verify llama-cpp-python
python3 -c "import llama_cpp; print('llama-cpp-python installed')"
```

---

## 📌 Notes
- NumPy (1.26.4), SciPy (1.11.4), and FilterPy (1.4.5) are installed via uv for high-speed installation and compatibility with Python 3.12.3.

- OpenCV and FFmpeg are built from source to enable CUDA support.

- The guide assumes an NVIDIA RTX 3000 GPU (CUDA Compute Capability 8.6).
  For other GPUs, update the CUDA_ARCH_BIN variable in INSTALL.md:

  7.5 for RTX 2000

  8.9 for RTX 4000

> ⚠️ Reboot is required after installing NVIDIA drivers.

---

## 🛠 Troubleshooting
- NVIDIA driver issues:
  Check logs with `journalctl -u nvidia*` or reinstall drivers.

- CUDA errors:
  Verify with `nvcc --version` and check environment variables.

- Python package conflicts:
  Recreate the virtual environment:

``` bash
rm -rf ~/venv
# Then rerun the setup steps
```
- Build failures (e.g., OpenCV):
  Check `make_log.txt` in the build directory or consult the official NVIDIA CUDA Installation Guide.

---

## 🤝 Contributing
Contributions are welcome! Feel free to submit issues or pull requests to suggest improvements or fix bugs.

---

## 📄 License
This project is licensed under the MIT License.

---

## 🙏 Acknowledgments
- [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)

- [OpenCV](https://opencv.org/)

- [FFmpeg](https://ffmpeg.org/)

- [PyTorch](https://pytorch.org/)

- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)
