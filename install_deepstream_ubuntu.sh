#!/bin/bash

# DeepStream Installation Script for Ubuntu Server
# This script installs DeepStream and its dependencies

echo "🚀 Installing DeepStream on Ubuntu Server..."

# Update system
echo "📦 Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install CUDA dependencies
echo "🔧 Installing CUDA dependencies..."
sudo apt install -y build-essential cmake pkg-config unzip yasm checkinstall libjpeg-dev libpng-dev libtiff-dev libavcodec-dev libavformat-dev libswscale-dev libavresample-dev libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libxvidcore-dev x264 libx264-dev libfaac-dev libmp3lame-dev libtheora-dev libfaac-dev libmp3lame-dev libvorbis-dev libopencore-amrnb-dev libopencore-amrwb-dev libdc1394-22 libdc1394-22-dev libxine2-dev libv4l-dev v4l-utils libgtk-3-dev libtbb-dev libatlas-base-dev gfortran libgtkglext1-dev libgstreamer-plugins-bad1.0-dev libgstreamer-plugins-good1.0-dev libgstreamer-plugins-ugly1.0-dev libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libgstreamer-plugins-bad1.0-dev libgstreamer-plugins-good1.0-dev libgstreamer-plugins-ugly1.0-dev libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev

# Install Python dependencies
echo "🐍 Installing Python dependencies..."
sudo apt install -y python3-pip python3-dev python3-venv
pip3 install --upgrade pip

# Install PyTorch with CUDA support
echo "🔥 Installing PyTorch with CUDA support..."
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install OpenCV with CUDA support
echo "📹 Installing OpenCV with CUDA support..."
pip3 install opencv-python opencv-contrib-python

# Install DeepStream Python bindings
echo "🔍 Installing DeepStream Python bindings..."
pip3 install pyds

# Install additional AI dependencies
echo "🧠 Installing AI model dependencies..."
pip3 install transformers accelerate sentence-transformers faiss-gpu

# Create DeepStream configuration directory
echo "📁 Creating DeepStream configuration directory..."
sudo mkdir -p /opt/nvidia/deepstream/deepstream-6.4
sudo chmod 755 /opt/nvidia/deepstream/deepstream-6.4

# Set environment variables
echo "🌍 Setting environment variables..."
echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.bashrc
echo 'export PATH=$PATH:$CUDA_HOME/bin' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64' >> ~/.bashrc
echo 'export DEEPSTREAM_HOME=/opt/nvidia/deepstream/deepstream-6.4' >> ~/.bashrc
echo 'export PATH=$PATH:$DEEPSTREAM_HOME/bin' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$DEEPSTREAM_HOME/lib' >> ~/.bashrc

# Source the updated bashrc
source ~/.bashrc

echo "✅ DeepStream installation completed!"
echo "🔧 Please restart your terminal or run: source ~/.bashrc"
echo "🚀 You can now run the AI Video Detective with DeepStream support!"

# Check CUDA installation
echo "🔍 Checking CUDA installation..."
nvidia-smi
nvcc --version

echo "🎯 DeepStream installation script completed successfully!"
