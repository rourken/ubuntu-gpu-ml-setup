# Installation Commands for Computer Vision and LLM Environment on Ubuntu 24.04.2 LTS

This guide provides step-by-step commands to set up a development environment for computer vision and large language models (LLM) on Ubuntu 24.04.2 LTS with Nvidia GPU support. All packages are installed in a virtual environment at `$HOME/cv_venv`, with a base path for source files at `$HOME/soft`.

## Prerequisites
- **OS**: Ubuntu 24.04.2 LTS
- **Hardware**: Nvidia GPU (e.g., RTX 3000 with CUDA compute capability 8.6)
- **Disk Space**: At least 50 GB free
- **RAM**: 16 GB or more recommended
- **Internet**: Required for downloading packages
- **Permissions**: Sudo access

## Installation Steps

### 1. Update System and Install Base Packages
Update the system and install essential packages for development, Python, and multimedia codecs.

```bash
sudo apt-get update && sudo apt-get upgrade -y
sudo apt-get install -y \
    build-essential cmake git autoconf automake libtool pkg-config \
    python3 python3-dev python3-pip python3-venv python3-packaging \
    libass-dev libfreetype6-dev libgnutls28-dev libsdl2-dev libva-dev libvdpau-dev libvorbis-dev \
    libxcb1-dev libxcb-shm0-dev libxcb-xfixes0-dev zlib1g-dev nasm yasm libx264-dev \
    libx265-dev libvpx-dev libfdk-aac-dev libmp3lame-dev libopus-dev libjpeg-dev \
    libpng-dev libtiff-dev libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \
    libxvidcore-dev x264 portaudio19-dev v4l-utils libunistring-dev libaom-dev libdav1d-dev \
    libgtk-3-dev libgtk2.0-dev qtbase5-dev qtchooser qt5-qmake qtbase5-dev-tools pyqt5-dev-tools \
    libatlas-base-dev gfortran libtbb-dev libnuma-dev \
    texinfo wget unzip libprotobuf-dev protobuf-compiler libjpeg8-dev libfaac-dev libtheora-dev \
    libopencore-amrnb-dev libopencore-amrwb-dev gcc screen ssh libomp-dev gnupg net-tools curl neofetch htop iproute2 dnsutils traceroute
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

### 3. Remove Old Nvidia Drivers and CUDA (if installed)
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

### 4. Install CUDA 12.6 and Nvidia Drivers
Install CUDA 12.6, Nvidia drivers, and cuDNN.
```bash

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-6 cuda-drivers cudnn-cuda-12
sudo ldconfig
rm cuda-keyring_1.1-1_all.deb
```

--- 

### 5. Reboot the System
Reboot to ensure Nvidia drivers are loaded correctly.
```bash

sudo reboot
```

--- 

### 6. Prevent Automatic Updates for Nvidia and CUDA
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

### 7. Set CUDA Environment Variables
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

### 8. Verify System Status
Check the system configuration and Nvidia setup.
```bash

neofetch
nvidia-smi
nvcc --version
uname -r
dkms status
lsmod | grep nvidia
```

--- 

### 9. Install OpenBLAS
Build and install OpenBLAS for optimized linear algebra computations.
``` bash

cd $HOME/soft
git clone https://github.com/OpenMathLib/OpenBLAS.git
cd OpenBLAS
make -j$(nproc)
sudo make install PREFIX=/usr/local
sudo ldconfig
cd $HOME/soft
```

--- 

### 10. Create and Activate Virtual Environment
Set up a Python virtual environment.
``` bash

sudo apt-get install -y python3 python3-dev python3-pip python3-venv python3-packaging
pip3 install --upgrade pip
python3 -m venv $HOME/cv_venv
source $HOME/cv_venv/bin/activate
```

--- 

### 11. Install NumPy
Install NumPy 1.26.4 (recommended for Python 3.12.3 compatibility).
``` bash

pip install numpy==1.26.4
python3 -c "import numpy as np; np.show_config()"
```

--- 

### 12. Install SciPy
Install SciPy 1.11.4 and its dependencies.
``` bash

pip install pybind11==2.10.4 pythran==0.14.0
pip install scipy==1.11.4
python3 -c "import scipy; scipy.show_config()"
```

--- 

### 13. Install FilterPy and Additional Python Packages
Install FilterPy and other required Python packages.
```bash

pip install filterpy==1.4.5
pip install -U setuptools matplotlib scikit-image Pillow fonttools PyWavelets tifffile pika requests lap fuzzywuzzy python-logstash python-Levenshtein
python3 -c "import numpy as np; from filterpy.kalman import KalmanFilter; from filterpy.common import Q_discrete_white_noise; print(KalmanFilter(dim_x=2, dim_z=1))"
```

--- 

### 14. Install FFmpeg with CUDA Support
Build FFmpeg with Nvidia GPU acceleration.
``` bash

cd $HOME/soft
git clone https://github.com/FFmpeg/nv-codec-headers.git -b sdk/12.2
cd nv-codec-headers
make -j$(nproc)
sudo make install
sudo ldconfig
cd $HOME/soft

git clone https://github.com/FFmpeg/FFmpeg.git -b n6.1.2
cd FFmpeg
export PATH=/usr/local/cuda-12.6/bin:$HOME/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH
export CFLAGS="-fPIC"
export CXXFLAGS="-fPIC"
./configure \
    --prefix=/usr/local \
    --enable-shared \
    --enable-gpl \
    --enable-libass \
    --enable-libfdk-aac \
    --enable-libfreetype \
    --enable-libmp3lame \
    --enable-libopus \
    --enable-libvorbis \
    --enable-libvpx \
    --enable-libx264 \
    --enable-libx265 \
    --enable-nonfree \
    --enable-cuda-nvcc \
    --enable-cuvid \
    --enable-nvenc \
    --enable-pic \
    --extra-cflags=-I/usr/local/cuda-12.6/include \
    --extra-ldflags=-L/usr/local/cuda-12.6/lib64
make -j$(nproc)
sudo make install
sudo ldconfig
cd $HOME/soft
```

--- 

### 15. Install OpenCV with CUDA Support
Build OpenCV 4.11.0 with CUDA and install it in the virtual environment.
```bash

cd $HOME/soft
git clone https://github.com/opencv/opencv.git -b 4.11.0
git clone https://github.com/opencv/opencv_contrib.git -b 4.11.0
cd opencv
mkdir build && cd build
cmake \
    -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D INSTALL_PYTHON_EXAMPLES=OFF \
    -D INSTALL_C_EXAMPLES=OFF \
    -D OPENCV_ENABLE_NONFREE=ON \
    -D WITH_TBB=ON \
    -D WITH_V4L=ON \
    -D WITH_QT=ON \
    -D WITH_OPENGL=ON \
    -D ENABLE_FAST_MATH=1 \
    -D CUDA_FAST_MATH=1 \
    -D WITH_CUBLAS=1 \
    -D WITH_CUDA=ON \
    -D WITH_CUDNN=ON \
    -D OPENCV_DNN_CUDA=ON \
    -D OPENCV_PC_FILE_NAME=opencv.pc \
    -D OPENCV_GENERATE_PKGCONFIG=ON \
    -D CUDA_ARCH_BIN="8.6" \
    -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
    -D BUILD_EXAMPLES=OFF \
    -D WITH_GSTREAMER=OFF \
    -D PYTHON_EXECUTABLE=$HOME/cv_venv/bin/python \
    -D OPENCV_PYTHON3_INSTALL_PATH=$HOME/cv_venv/lib/python3.12/site-packages \
    ..
make -j$(nproc)
sudo make install
sudo ldconfig
cd $HOME/soft
python3 -c "import cv2; print(cv2.getBuildInformation())"
```

--- 

### 16. Install PyTorch with CUDA
Install PyTorch with CUDA 12.6 support.
```bash

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
python3 -c "import torch; print(torch.cuda.is_available())"
```

--- 

### 17. Install llama-cpp-python with CuBlas
Install llama-cpp-python with CUDA and CuBlas support.
```bash

CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=86 -DGGML_CUDA_F16=ON -DGGML_CUDA_FORCE_CUBLAS=ON" \
FORCE_CMAKE=1 pip install --no-cache-dir llama-cpp-python
python3 -c "import llama_cpp; print('llama-cpp-python installed')"
```

--- 

### 18. Install Additional Libraries
Install additional Python libraries for extended functionality.
``` bash

pip install --no-cache-dir werkzeug uvicorn fastapi pydub matplotlib sounddevice librosa deskew python-multipart PyYAML ultralytics
pip install git+https://github.com/SiggiGue/pyfilterbank.git
pip uninstall -y opencv-python || true
```

--- 

### Post-Installation
Activate Virtual Environment: Run source `$HOME/cv_venv/bin/activate` in every new terminal session.

> Verify Setup: Re-run verification commands from Step 8 and Steps 11–17 to ensure all components are working.

--- 

### Notes
- Reboot after Step 5 is critical to load Nvidia drivers.

- NumPy, SciPy, and FilterPy are installed via pip for Python 3.12.3 compatibility.

- OpenCV and FFmpeg are built from source to enable CUDA support.

- CUDA_ARCH_BIN=8.6 is set for RTX 3000 GPUs. Adjust to 7.5 for RTX 2000 or 8.9 for RTX 4000 if needed.




