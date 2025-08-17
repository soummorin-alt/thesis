# Real-Time Vehicle Speed Estimation System

A complete monocular vehicle speed estimation system for head-mounted cameras, implementing automatic camera calibration, vehicle detection, tracking, and speed calculation based on state-of-the-art computer vision research.

## Features

🚗 **Real-time vehicle detection** using YOLOv8  
📹 **Automatic camera calibration** via vanishing point detection  
🎯 **Multi-object tracking** with ByteTrack integration  
⚡ **Optical flow enhancement** for improved accuracy  
📊 **Speed estimation** with ±5-6 km/h target accuracy  
🎨 **Real-time visualization** with comprehensive overlays  
💾 **Calibration persistence** and result export  

## Technical Approach

This system implements the methodologies from several key research papers:

- **Dubská & Herout (2014)**: Automatic camera calibration via vanishing points
- **Luvizon et al.**: License plate tracking for speed estimation  
- **Barros & Oliveira**: Deep optical flow with CNN regressors

### Pipeline Overview

1. **Vanishing Point Detection**: Automatic camera intrinsic/extrinsic parameter estimation
2. **Ground Plane Homography**: Mapping between image and real-world coordinates
3. **Vehicle Detection**: YOLOv8-based real-time detection
4. **Multi-Object Tracking**: ByteTrack for robust vehicle trajectories
5. **Speed Calculation**: Ground-plane projection with temporal smoothing
6. **Optical Flow Enhancement**: Dense flow for improved motion estimation

## Installation

### Quick Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd vehicle-speed-estimation

# Run the setup script
chmod +x quick_setup.sh
./quick_setup.sh

# Activate the environment
source venv/bin/activate

# Install additional dependencies
pip install -r requirements.txt
```

### Manual Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install PyTorch (adjust for your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install ultralytics opencv-python numpy scipy matplotlib
pip install scikit-image filterpy lap cython-bbox motmetrics
pip install loguru tqdm Pillow imageio

# Install ByteTrack (optional but recommended)
git clone https://github.com/ifzhang/ByteTrack.git
cd ByteTrack
pip install -r requirements.txt
python setup.py develop
cd ..
```

### Dataset Setup (Optional)

```bash
# Download publicly available datasets
python setup_datasets.py

# For KITTI and other datasets, follow the manual instructions printed
```

## Usage

### Real-Time Webcam Processing

```bash
# Basic usage with default webcam
python main_webcam.py

# Advanced usage with custom parameters
python main_webcam.py \
    --camera 0 \
    --camera_height 1.7 \
    --confidence 0.5 \
    --width 1280 \
    --height 720 \
    --save_video output.mp4 \
    --save_calibration calibration.json
```

**Webcam Controls:**
- `q` - Quit application
- `c` - Force recalibration  
- `s` - Save calibration
- `r` - Reset statistics
- `h` - Toggle help overlay

### Video File Processing

```bash
# Process a video file
python main_video.py input_video.mp4 \
    --output_video processed_output.mp4 \
    --save_results results.json \
    --display

# Batch processing with custom parameters
python main_video.py traffic_video.mp4 \
    --camera_height 2.0 \
    --confidence 0.6 \
    --load_calibration saved_calibration.json \
    --save_results detailed_results.json \
    --skip_frames 2
```

### Advanced Configuration

```bash
# High-accuracy mode with optical flow
python main_webcam.py \
    --camera_height 1.8 \
    --confidence 0.4 \
    --width 1920 \
    --height 1080

# Fast mode without optical flow
python main_webcam.py \
    --no_optical_flow \
    --confidence 0.6 \
    --width 640 \
    --height 480
```

## System Requirements

### Minimum Requirements
- Python 3.8+
- 4GB RAM
- CPU: Intel i5 or AMD Ryzen 5
- Webcam or video input

### Recommended Requirements
- Python 3.9+
- 8GB+ RAM
- GPU: NVIDIA GTX 1060 or better (for real-time processing)
- CUDA 11.8+ (for GPU acceleration)

### Performance Expectations
- **CPU-only**: 5-15 FPS (depending on resolution)
- **GPU-accelerated**: 15-30 FPS (1080p real-time capable)
- **Accuracy**: ±5-6 km/h target (±2-3 km/h achievable with good calibration)

## Calibration

The system performs automatic calibration using vanishing point detection:

### Automatic Calibration
1. Point camera at road scene with visible lane markings or vehicle edges
2. Ensure multiple vehicles are visible for scale estimation
3. System will automatically detect vanishing points and calibrate
4. Calibration typically completes within 30-100 frames

### Manual Calibration Tips
- **Camera Height**: Measure actual height above ground (default: 1.7m for eye level)
- **Scene Requirements**: Clear lane markings, multiple vehicles, good lighting
- **Avoid**: Heavy occlusion, very curved roads, extreme weather

### Calibration Persistence
```bash
# Save calibration for reuse
python main_webcam.py --save_calibration my_calibration.json

# Load existing calibration
python main_webcam.py --load_calibration my_calibration.json
```

## Accuracy and Validation

### Expected Performance
- **Target Accuracy**: ±5-6 km/h (as per feasibility analysis)
- **Achieved in Research**: 
  - Dubská et al.: ~2% error (±1.5 km/h at 75 km/h)
  - Barros & Oliveira: 85% within ±2-3 km/h
- **Real-world Factors**: Camera stability, calibration quality, scene complexity

### Validation with BrnoCompSpeed Dataset
```bash
# Process BrnoCompSpeed videos (if you have the dataset)
python main_video.py brno_video.avi \
    --save_results brno_results.json \
    --camera_height 4.0  # Adjust for dataset camera height
```

## Troubleshooting

### Common Issues

**1. Poor Detection Performance**
```bash
# Lower confidence threshold
python main_webcam.py --confidence 0.3

# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"
```

**2. Calibration Failures**
- Ensure clear lane markings are visible
- Check camera height parameter matches reality
- Verify multiple vehicles are present in scene

**3. Low FPS Performance**
```bash
# Reduce resolution
python main_webcam.py --width 640 --height 480

# Disable optical flow
python main_webcam.py --no_optical_flow
```

**4. ByteTrack Installation Issues**
```bash
# Use fallback simple tracker (automatic if ByteTrack unavailable)
# Or install manually:
pip install cython
pip install cython-bbox
```

### Debug Mode
```python
# Enable verbose logging in code
import logging
logging.basicConfig(level=logging.DEBUG)
```

## File Structure

```
vehicle-speed-estimation/
├── src/
│   ├── calibration/
│   │   ├── vanishing_point_detector.py
│   │   └── camera_calibrator.py
│   ├── detection/
│   │   └── vehicle_detector.py
│   ├── tracking/
│   │   └── byte_tracker.py
│   ├── flow/
│   │   └── optical_flow.py
│   └── speed_estimator.py
├── main_webcam.py          # Real-time webcam processing
├── main_video.py           # Video file processing
├── setup_datasets.py       # Dataset download utility
├── quick_setup.sh          # Quick installation script
├── requirements.txt        # Python dependencies
└── README.md
```

## Research Papers and References

1. **Dubská, M., & Herout, A.** (2014). "Automatic Camera Calibration for Traffic Understanding." BMVA.
2. **Luvizon, D., et al.** "A Video-Based System for Vehicle Speed Measurement in Urban Roadways."
3. **Barros, A., & Oliveira, M.** "Deep Speed Estimation from Synthetic and Monocular Data."
4. **Sochor, J., et al.** "BrnoCompSpeed: Review of Traffic Camera Calibration and Comprehensive Dataset for Monocular Speed Measurement."

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- ByteTrack team for excellent multi-object tracking
- Ultralytics for YOLOv8 implementation
- Research authors for foundational methodologies
- BrnoCompSpeed dataset creators

## Citation

If you use this system in your research, please cite:

```bibtex
@software{vehicle_speed_estimation_2024,
  title={Real-Time Vehicle Speed Estimation System},
  author={Your Name},
  year={2024},
  url={https://github.com/your-username/vehicle-speed-estimation}
}
```