#!/bin/bash
# Quick Setup for Vehicle Speed Estimation System

echo "Setting up Vehicle Speed Estimation System..."

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install essential packages
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics  # YOLOv8
pip install opencv-python
pip install numpy scipy matplotlib
pip install wget

# Install ByteTrack
pip install yolox
git clone https://github.com/ifzhang/ByteTrack.git
cd ByteTrack
pip install -r requirements.txt
python setup.py develop
cd ..

# Install RAFT for optical flow (optional - can use OpenCV instead)
pip install timm

# Download pre-trained models
mkdir -p models
cd models

# YOLOv8 models (will auto-download on first use)
echo "YOLOv8 models will auto-download when first used"

# ByteTrack model
wget https://github.com/ifzhang/ByteTrack/releases/download/v0.1.0/bytetrack_x_mot17.pth.tar

cd ..

echo "Setup complete! Run: source venv/bin/activate"
echo "Then run the dataset setup: python setup_datasets.py"