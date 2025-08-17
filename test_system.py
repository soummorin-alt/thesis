#!/usr/bin/env python3
"""
Test script for Vehicle Speed Estimation System
Verifies that all components work correctly
"""

import sys
import os
import numpy as np
import cv2

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    
    try:
        from src.calibration.vanishing_point_detector import VanishingPointDetector
        print("✓ VanishingPointDetector imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import VanishingPointDetector: {e}")
        return False
    
    try:
        from src.calibration.camera_calibrator import CameraCalibrator
        print("✓ CameraCalibrator imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import CameraCalibrator: {e}")
        return False
    
    try:
        from src.detection.vehicle_detector import VehicleDetector
        print("✓ VehicleDetector imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import VehicleDetector: {e}")
        return False
    
    try:
        from src.tracking.byte_tracker import VehicleTracker
        print("✓ VehicleTracker imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import VehicleTracker: {e}")
        return False
    
    try:
        from src.flow.optical_flow import OpticalFlowEstimator
        print("✓ OpticalFlowEstimator imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import OpticalFlowEstimator: {e}")
        return False
    
    try:
        from src.speed_estimator import VehicleSpeedEstimator
        print("✓ VehicleSpeedEstimator imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import VehicleSpeedEstimator: {e}")
        return False
    
    return True

def test_basic_functionality():
    """Test basic functionality with dummy data"""
    print("\nTesting basic functionality...")
    
    try:
        from src.speed_estimator import VehicleSpeedEstimator
        
        # Create a dummy estimator
        estimator = VehicleSpeedEstimator(
            camera_height=1.7,
            confidence_threshold=0.5,
            use_optical_flow=False  # Disable for testing
        )
        print("✓ VehicleSpeedEstimator created successfully")
        
        # Create a dummy frame
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Add some pattern to make it more realistic
        cv2.rectangle(dummy_frame, (100, 200), (200, 300), (255, 255, 255), -1)
        cv2.rectangle(dummy_frame, (400, 250), (500, 350), (128, 128, 128), -1)
        
        # Process the frame
        result = estimator.process_frame(dummy_frame)
        print("✓ Frame processed successfully")
        
        # Check result structure
        required_keys = ['frame_id', 'timestamp', 'detections', 'tracked_vehicles', 
                        'speeds', 'calibration_status', 'processing_time', 'fps']
        
        for key in required_keys:
            if key not in result:
                print(f"✗ Missing key in result: {key}")
                return False
        
        print("✓ Result structure is correct")
        print(f"  - Frame ID: {result['frame_id']}")
        print(f"  - Detections: {len(result['detections'])}")
        print(f"  - Tracked vehicles: {len(result['tracked_vehicles'])}")
        print(f"  - Processing time: {result['processing_time']:.3f}s")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in basic functionality test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_detector():
    """Test vehicle detector specifically"""
    print("\nTesting vehicle detector...")
    
    try:
        from src.detection.vehicle_detector import VehicleDetector
        
        detector = VehicleDetector(confidence_threshold=0.5)
        print("✓ VehicleDetector created successfully")
        
        # Create a more realistic test image
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        detections = detector.detect_vehicles(test_image)
        print(f"✓ Detection completed, found {len(detections)} vehicles")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in detector test: {e}")
        return False

def test_calibrator():
    """Test camera calibrator"""
    print("\nTesting camera calibrator...")
    
    try:
        from src.calibration.camera_calibrator import CameraCalibrator
        
        calibrator = CameraCalibrator(
            image_width=640,
            image_height=480,
            assumed_camera_height=1.7
        )
        print("✓ CameraCalibrator created successfully")
        
        # Test principal point estimation
        pp = calibrator.estimate_principal_point()
        print(f"✓ Principal point estimated: ({pp[0]:.1f}, {pp[1]:.1f})")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in calibrator test: {e}")
        return False

def test_dependencies():
    """Test that all required dependencies are available"""
    print("\nTesting dependencies...")
    
    dependencies = [
        ('cv2', 'OpenCV'),
        ('numpy', 'NumPy'),
        ('torch', 'PyTorch'),
        ('ultralytics', 'Ultralytics (YOLOv8)'),
        ('sklearn', 'Scikit-learn'),
    ]
    
    all_good = True
    
    for module, name in dependencies:
        try:
            __import__(module)
            print(f"✓ {name} available")
        except ImportError:
            print(f"✗ {name} not available")
            all_good = False
    
    # Test optional dependencies
    optional_deps = [
        ('yolox', 'YOLOX (ByteTrack)'),
    ]
    
    for module, name in optional_deps:
        try:
            __import__(module)
            print(f"✓ {name} available (optional)")
        except ImportError:
            print(f"⚠ {name} not available (optional, will use fallback)")
    
    return all_good

def main():
    """Run all tests"""
    print("Vehicle Speed Estimation System - Test Suite")
    print("=" * 50)
    
    tests = [
        ("Dependencies", test_dependencies),
        ("Imports", test_imports),
        ("Calibrator", test_calibrator),
        ("Detector", test_detector),
        ("Basic Functionality", test_basic_functionality),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{test_name} Test")
        print("-" * 30)
        
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"{test_name:20} - {status}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! System is ready to use.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
        return 1

if __name__ == '__main__':
    sys.exit(main())