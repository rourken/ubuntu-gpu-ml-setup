# Installation Commands for Computer Vision and LLM Environment on Ubuntu 24.04.3 LTS

This guide provides step-by-step commands to set up a development environment for computer vision and large language models (LLM) on Ubuntu 24.04.3 LTS with Nvidia GPU support. All packages are installed in a virtual environment at `$HOME/venv`, with a base path for source files at `$HOME/soft`.

## Prerequisites
- **OS**: Ubuntu 24.04.3 LTS
- **Hardware**: Nvidia GPU (e.g., RTX 3000 with CUDA compute capability 8.6)
- **Disk Space**: At least 50 GB free
- **RAM**: 16 GB or more recommended
- **Internet**: Required for downloading packages
- **Permissions**: Sudo access

## Installation Steps

## === TARGET ARCHITECTURE CONFIGURATION ===
Set architecture specific variables
86 = Nvidia RTX 3000 series (Ampere), excluding RTX 3050

```bash 
export ARCH=86
export VIRTUAL_ENV=$HOME/venv

```

### 1. Update System and Install Base Packages
Update the system and install essential packages for development, Python, and multimedia codecs.

```bash
sudo apt-get install -y software-properties-common
sudo add-apt-repository -y multiverse
sudo apt-get update && sudo apt-get upgrade -y
sudo apt-get install -y --no-install-recommends \
    software-properties-common build-essential gcc g++ gfortran \
    autoconf automake libtool pkg-config cmake git yasm nasm wget \
    curl nano python3 python3-dev python3-pip python3-venv python3-packaging \
    libass-dev libfreetype6-dev libgnutls28-dev libsdl2-dev libva-dev libvdpau-dev \
    libvorbis-dev libxcb1-dev libxcb-shm0-dev libxcb-xfixes0-dev zlib1g-dev libx264-dev \
    libx265-dev libvpx-dev libmp3lame-dev libopus-dev libjpeg-dev libpng-dev libtiff-dev \
    libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libgl1-mesa-dev libxvidcore-dev \
    x264 portaudio19-dev v4l-utils libunistring-dev libaom-dev libdav1d-dev libfdk-aac-dev \
    libgtk-3-dev libgtk2.0-dev qtbase5-dev qtchooser qt5-qmake qtbase5-dev-tools pyqt5-dev-tools \
    libatlas-base-dev libtbb-dev libnuma-dev texinfo libprotobuf-dev protobuf-compiler libjpeg8-dev \
    libfaac-dev libtheora-dev libopencore-amrnb-dev libopencore-amrwb-dev screen ssh libomp-dev gnupg \
    net-tools curl neofetch htop btop nvtop iproute2 dnsutils traceroute jq duf bat
sudo ldconfig
sudo apt-get autoremove -y && sudo apt-get clean
```

--- 

### 2. Create Base Directory
Create a directory for source files and builds.

```bash
mkdir -p $HOME/soft
cd $HOME/soft
```

---

### 3. Install UV
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env || export PATH="$HOME/.cargo/bin:$PATH"
sudo ldconfig
```


---

### 4. Remove Old Nvidia Drivers and CUDA (if installed)
Remove any existing Nvidia drivers and CUDA installations to avoid conflicts.
```bash
if dpkg -l | grep -q nvidia; then
    sudo apt-get purge -y cuda-keyring '^nvidia-.*' '^libnvidia-.*' '^cuda.*' '^cudnn.*' '^libcudnn.*'
    sudo apt-get autoremove -y && sudo apt-get autoclean -y
    sudo rm -rf /usr/local/cuda* /usr/local/nvidia* /etc/apt/sources.list.d/cuda* /etc/apt/sources.list.d/nvidia*
fi
sudo apt-get update
```

--- 

### 5. Install CUDA 12.6 and Nvidia Drivers
Install CUDA 12.6, Nvidia drivers, and cuDNN.
```bash

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-6 cuda-drivers cudnn-cuda-12 libcudnn9-dev-cuda-12
sudo ldconfig
rm cuda-keyring_1.1-1_all.deb
```

--- 

### 6. Reboot the System
Reboot to ensure Nvidia drivers are loaded correctly.
```bash

sudo reboot
```

--- 

### 7. Prevent Automatic Updates for Nvidia and CUDA
Block automatic updates to maintain stability.
```bash

echo 'Package: nvidia*
Pin: release *
Pin-Priority: -1

Package: cuda*
Pin: release *
Pin-Priority: -1' | sudo tee /etc/apt/preferences.d/cuda-repository-pin-noupdate
```

--- 

### 8. Set CUDA Environment Variables
Configure environment variables for CUDA.
``` bash

echo 'export CUDA_HOME=/usr/local/cuda-12.6' >> ~/.bashrc
echo 'export CUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME' >> ~/.bashrc
echo 'export PATH=$CUDA_HOME/bin${PATH:+:${PATH}}' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc
source ~/.bashrc
sudo ldconfig
```

--- 

### 9. Verify System Status
Check the system configuration and Nvidia setup.
```bash

neofetch
nvidia-smi
nvcc --version
uv --version
uname -r
dkms status
lsmod | grep nvidia
```

--- 

### 10. Install OpenBLAS v0.3.30
Build and install OpenBLAS for optimized linear algebra computations.
``` bash
export OPENBLAS_BUILD_OPTS="DYNAMIC_ARCH=1 USE_OPENMP=1 NO_AFFINITY=1 FC=gfortran PREFIX=/usr/local"
cd $HOME/soft
git clone --depth 1 --branch v0.3.30 https://github.com/OpenMathLib/OpenBLAS.git 
cd OpenBLAS
make ${OPENBLAS_BUILD_OPTS} -j$(nproc)
sudo make install ${OPENBLAS_BUILD_OPTS}
sudo ldconfig
cd $HOME/soft
rm -rf OpenBLAS
```
# --- Configure OpenBLAS paths for subsequent builds ---
```bash 
export LD_LIBRARY_PATH=/usr/local/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export LIBRARY_PATH=/usr/local/lib${LIBRARY_PATH:+:${LIBRARY_PATH}}
export CPATH=/usr/local/include${CPATH:+:${CPATH}}

echo "=== Check OpenBLAS ===" && \
    ls -lh /usr/local/lib/libopenblas* && \
    ls -lh /usr/local/include/cblas.h && \
    echo "OpenBLAS install done!"
```
--- 

### 11. Create and Activate Virtual Environment
Set up a Python virtual environment.
``` bash

uv venv $VIRTUAL_ENV --seed --clear
export PATH="${VIRTUAL_ENV}/bin:${PATH}"
```

--- 

### 12. Build NumPy and SciPy by source with OpenBLAS
Install NumPy 1.26.4 (recommended for Python 3.12.3 compatibility).
``` bash
cd $HOME/soft
mkdir -p ./numpy-build
printf "[openblas]\n\
libraries = openblas\n\
library_dirs = /usr/local/lib\n\
include_dirs = /usr/local/include\n" > site.cfg && \
    uv pip install -U pip setuptools wheel Cython pybind11 pythran meson meson-python pyyaml && \
    uv pip install --no-binary=numpy "numpy==1.26.4" && \
    uv pip install --no-binary=scipy "scipy==1.11.4" && \
    uv run python - << 'EOF'
import numpy as np, scipy
print("NumPy:", np.__version__)
print("SciPy:", scipy.__version__)
EOF

rm -rf ./numpy-build
```

--- 

### 13. Install FilterPy and Additional Python Packages
Install FilterPy and other required Python packages.
```bash

uv pip install --no-cache-dir matplotlib scikit_image Pillow \
    fonttools PyWavelets tifffile pika requests lap fuzzywuzzy \
    python-Levenshtein tqdm polars ultralytics-thop \
    psutil py-cpuinfo pandas seaborn
    

# install filterpy + path
uv pip install filterpy==1.4.5 && \
    uv run python - << 'EOF'
import os, site
site_dir = next(p for p in site.getsitepackages() if p.endswith("site-packages"))
stats_path = os.path.join(site_dir, "filterpy", "stats", "stats.py")

txt = open(stats_path, "r", encoding="utf-8").read()
marker = "except TypeError:"
if marker in txt:
    open(stats_path, "w", encoding="utf-8").write(txt.replace(marker, "except Exception:", 1))
    print("Patched filterpy.stats.stats: 'except TypeError' -> 'except Exception'")
else:
    print("Patch marker not found in filterpy.stats.stats (possibly already fixed)")

from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
kf = KalmanFilter(dim_x=2, dim_z=1)
print("KalmanFilter created OK:", kf)
EOF
```

--- 

### 14. Install FFmpeg with CUDA Support
#### Build NVIDIA Video Codec Headers (13.0.19.0)
Build FFmpeg with Nvidia GPU acceleration.
``` bash

cd $HOME/soft
git clone --depth 1 --branch n13.0.19.0 https://github.com/FFmpeg/nv-codec-headers.git
cd nv-codec-headers
make PREFIX=/usr/local
sudo make PREFIX=/usr/local install
sudo ldconfig
cd $HOME/soft
```

#### Build FFmpeg 7.1.2 with NVENC/NVDEC
```bash
cd $HOME/soft
git clone --depth 1 --branch n7.1.2 https://github.com/FFmpeg/FFmpeg.git
cd FFmpeg
# export PATH=/usr/local/cuda-12.6/bin:$HOME/bin:$PATH
./configure \
    --prefix=/usr/local \
    --enable-shared \
    --enable-gpl \
    --enable-version3 \
    --enable-libass \
    --enable-libfdk-aac \
    --enable-libfreetype \
    --enable-libmp3lame \
    --enable-libopus \
    --enable-libvorbis \
    --enable-libvpx \
    --enable-libx264 \
    --enable-libx265 \
    --enable-pic \
    --enable-nonfree \
    --enable-cuda-nvcc \
    --enable-cuvid \
    --enable-nvenc \
    --nvccflags="-gencode arch=compute_${ARCH},code=sm_${ARCH} -O2" \
    --extra-cflags="-I/usr/local/cuda/include -I/usr/local/include" \
    --extra-ldflags="-L/usr/local/cuda/lib64 -L/usr/local/lib"
make -j$(nproc)
sudo make install
sudo ldconfig
cd $HOME/soft
rm -rf FFmpeg

echo "=== Check install FFmpeg ===" && \
    ffmpeg -version && \
    echo "--- NVENC codecs: ---" && \
    ffmpeg -hide_banner -encoders 2>/dev/null | grep nvenc || true && \
    echo "--- NVDEC/CUVID: ---" && \
    ffmpeg -hide_banner -decoders 2>/dev/null | grep cuvid || true && \
    echo "FFmpeg build with CUDA successfully"
```

--- 

### 15. Install OpenCV 4.12.0 with CUDA Support
Build OpenCV 4.12.0 with CUDA and install it in the virtual environment.
```bash

cd $HOME/soft
git clone --depth 1 --branch 4.12.0 https://github.com/opencv/opencv.git
git clone --depth 1 --branch 4.12.0 https://github.com/opencv/opencv_contrib.git
cd opencv
mkdir -p build && cd build
cmake -S .. -B . \
    -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
    -D WITH_FFMPEG=ON \
    -D WITH_CUDA=ON \
    -D WITH_CUDNN=ON \
    -D OPENCV_DNN_CUDA=ON \
    -D CUDA_ARCH_BIN=${ARCH} \
    -D ENABLE_FAST_MATH=ON \
    -D CUDA_FAST_MATH=ON \
    -D WITH_CUBLAS=ON \
    -D BLAS_LIBRARIES=/usr/local/lib/libopenblas.so \
    -D BLAS_INCLUDE_DIR=/usr/local/include \
    -D WITH_TBB=ON \
    -D WITH_V4L=ON \
    -D WITH_QT=ON \
    -D WITH_OPENGL=ON \
    -D BUILD_EXAMPLES=OFF \
    -D BUILD_TESTS=OFF \
    -D BUILD_PERF_TESTS=OFF \
    -D WITH_GTK=OFF \
    -D WITH_GSTREAMER=OFF \
    -D OPENCV_GENERATE_PKGCONFIG=ON \
    -D INSTALL_PYTHON_EXAMPLES=OFF \
    -D INSTALL_C_EXAMPLES=OFF \
    -D OPENCV_ENABLE_NONFREE=ON \
    -D OPENCV_PC_FILE_NAME=opencv4.pc \
    -D PYTHON3_EXECUTABLE=$VIRTUAL_ENV/bin/python3 \
    -D PYTHON3_INCLUDE_DIR=/usr/include/python3.12 \
    -D PYTHON3_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.12.so \
    -D OPENCV_PYTHON3_INSTALL_PATH=$VIRTUAL_ENV/lib/python3.12/site-packages \
    -D BUILD_opencv_python3=ON \
    -D BUILD_opencv_python2=OFF
make -j$(nproc)
sudo make install
sudo ldconfig
cd $HOME/soft
rm -rf opencv opencv_contrib

echo "=== Check OpenCV Installation ===" && \
    uv run python -c "import cv2; print(f'OpenCV Version: {cv2.__version__}')" && \
    echo "OpenCV installed successfully"
```

--- 

### 16. Install PyTorch, Ultralytics with CUDA
Install PyTorch with CUDA 12.6 support.
```bash
UV_TORCH_BACKEND=cu126

uv pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
python3 -c "import torch; print(torch.cuda.is_available())"

uv pip install --no-deps ultralytics==8.3.235
```

```bash
echo "=== FINAL CHECK ===" && \
    uv run python -c "import numpy; print(f'✓ NumPy: {numpy.__version__}')" && \
    uv run python -c "import torch; print(f'✓ PyTorch: {torch.__version__}'); print(f'✓ CUDA: {torch.cuda.is_available()}'); print(f'✓ CUDA Version: {torch.version.cuda}')" && \
    uv run python -c "import cv2; print(f'✓ OpenCV: {cv2.__version__}'); print(f'✓ OpenCV CUDA devices: {cv2.cuda.getCudaEnabledDeviceCount()}')" && \
    uv run python -c "import ultralytics; print(f'✓ Ultralytics: {ultralytics.__version__}')" && \
    uv run python -c "from filterpy.kalman import KalmanFilter; print('✓ KalmanFilter sanity check OK')" && \
    echo "=== ALL MODULES LOADED SUCCESSFULLY ==="
```

--- 

### 17. Install llama-cpp-python with CuBlas
Install llama-cpp-python with CUDA and CuBlas support.
```bash
CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=${ARCH} -DGGML_CUDA_F16=ON -DGGML_CUDA_FORCE_CUBLAS=ON" \
FORCE_CMAKE=1 uv pip install --no-cache-dir llama-cpp-python
echo "Checking installation..."
python3 -c "
import llama_cpp
print('✅ llama-cpp-python installed')
print(f'Version: {llama_cpp.__version__}')
print(f'CUDA Support: {\"available\" if llama_cpp.llama_cpp.llama_supports_gpu_offload() else \"not available\"}')"
```

--- 

### Post-Installation
Activate Virtual Environment: Run source `$HOME/venv/bin/activate` in every new terminal session.

> Verify Setup: Re-run verification commands from Step 8 and Steps 11–17 to ensure all components are working.

--- 

### Notes
- Reboot after Step 5 is critical to load Nvidia drivers.

- NumPy, SciPy, and FilterPy are installed via pip for Python 3.12.3 compatibility.

- OpenCV and FFmpeg are built from source to enable CUDA support.

- CUDA_ARCH_BIN=8.6 is set for RTX 3000 GPUs. Adjust to 7.5 for RTX 2000 or 8.9 for RTX 4000 if needed.
