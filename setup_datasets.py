#!/usr/bin/env python3
"""
Dataset Setup Script for Vehicle Speed Estimation
Downloads and organizes publicly available datasets
"""

import os
import wget
import zipfile
import tarfile
from pathlib import Path
import argparse

def create_directory(path):
    """Create directory if it doesn't exist"""
    Path(path).mkdir(parents=True, exist_ok=True)

def download_coco(data_dir):
    """Download COCO dataset for vehicle detection"""
    print("Downloading COCO 2017 dataset...")
    coco_dir = os.path.join(data_dir, "coco")
    create_directory(coco_dir)
    
    urls = [
        "http://images.cocodataset.org/zips/train2017.zip",
        "http://images.cocodataset.org/zips/val2017.zip", 
        "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    ]
    
    for url in urls:
        filename = os.path.join(coco_dir, url.split('/')[-1])
        if not os.path.exists(filename):
            print(f"Downloading {url}...")
            wget.download(url, filename)
            print(f"\nExtracting {filename}...")
            with zipfile.ZipFile(filename, 'r') as zip_ref:
                zip_ref.extractall(coco_dir)
        else:
            print(f"Already exists: {filename}")

def download_kitti(data_dir):
    """Download KITTI dataset"""
    print("Downloading KITTI Object Detection dataset...")
    kitti_dir = os.path.join(data_dir, "kitti")
    create_directory(kitti_dir)
    
    urls = [
        "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip",
        "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip"
    ]
    
    for url in urls:
        filename = os.path.join(kitti_dir, url.split('/')[-1])
        if not os.path.exists(filename):
            print(f"Downloading {url}...")
            wget.download(url, filename)
            print(f"\nExtracting {filename}...")
            with zipfile.ZipFile(filename, 'r') as zip_ref:
                zip_ref.extractall(kitti_dir)
        else:
            print(f"Already exists: {filename}")

def setup_synthetic_data_generators(data_dir):
    """Setup synthetic data generation tools"""
    print("Setting up synthetic data generators...")
    
    # CARLA setup instructions
    carla_dir = os.path.join(data_dir, "carla_setup")
    create_directory(carla_dir)
    
    setup_instructions = """
# CARLA Setup Instructions
# 1. Download CARLA from: https://github.com/carla-simulator/carla/releases
# 2. Extract and run: ./CarlaUE4.sh
# 3. Install Python API: pip install carla
# 4. Use our synthetic data generation scripts

# Alternative: Use pre-generated synthetic data
# Download from: https://drive.google.com/synthetic_vehicle_data
"""
    
    with open(os.path.join(carla_dir, "setup_instructions.txt"), "w") as f:
        f.write(setup_instructions)

def download_pretrained_models(data_dir):
    """Download pre-trained models"""
    print("Setting up pre-trained models...")
    models_dir = os.path.join(data_dir, "pretrained_models")
    create_directory(models_dir)
    
    model_urls = {
        "yolov8n.pt": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",
        "yolov8s.pt": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt",
    }
    
    for model_name, url in model_urls.items():
        model_path = os.path.join(models_dir, model_name)
        if not os.path.exists(model_path):
            print(f"Downloading {model_name}...")
            wget.download(url, model_path)
        else:
            print(f"Already exists: {model_name}")

def main():
    parser = argparse.ArgumentParser(description="Setup datasets for vehicle speed estimation")
    parser.add_argument("--data_dir", default="./datasets", help="Directory to store datasets")
    parser.add_argument("--skip_large", action="store_true", help="Skip large downloads (>5GB)")
    
    args = parser.parse_args()
    
    print(f"Setting up datasets in: {args.data_dir}")
    create_directory(args.data_dir)
    
    # Download smaller datasets first
    download_pretrained_models(args.data_dir)
    download_kitti(args.data_dir)
    
    if not args.skip_large:
        download_coco(args.data_dir)
    
    setup_synthetic_data_generators(args.data_dir)
    
    print("\n" + "="*50)
    print("DATASET SETUP COMPLETE!")
    print("="*50)
    print(f"Data directory: {args.data_dir}")
    print("\nManual steps required:")
    print("1. Register and download Cityscapes from: https://www.cityscapes-dataset.com/")
    print("2. You already have BrnoCompSpeed dataset ✓")
    print("3. For optical flow: Pre-trained RAFT models available via torchvision")
    print("4. For ByteTrack: pip install yolox")

if __name__ == "__main__":
    main()