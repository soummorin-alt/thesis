"""
Real-Time Vehicle Speed Estimation System
Integrates camera calibration, vehicle detection, tracking, and speed calculation
Based on Dubská & Herout, Luvizon et al., and Barros & Oliveira methodologies
"""

import cv2
import numpy as np
import time
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import json

from .calibration.camera_calibrator import CameraCalibrator
from .detection.vehicle_detector import VehicleDetector
from .tracking.byte_tracker import VehicleTracker
from .flow.optical_flow import OpticalFlowEstimator


class VehicleSpeedEstimator:
    """
    Complete real-time vehicle speed estimation system
    Performs automatic calibration and speed measurement from monocular video
    """
    
    def __init__(self, 
                 camera_height: float = 1.7,
                 confidence_threshold: float = 0.5,
                 flow_method: str = 'farneback',
                 use_optical_flow: bool = True,
                 calibration_frames: int = 30):
        """
        Initialize the vehicle speed estimation system
        
        Args:
            camera_height: Camera height above ground in meters (eye level)
            confidence_threshold: Detection confidence threshold
            flow_method: Optical flow method ('farneback', 'lucas_kanade', 'raft')
            use_optical_flow: Whether to use optical flow for enhanced tracking
            calibration_frames: Number of frames to use for initial calibration
        """
        self.camera_height = camera_height
        self.confidence_threshold = confidence_threshold
        self.use_optical_flow = use_optical_flow
        self.calibration_frames = calibration_frames
        
        # Initialize components (will be set when video source is known)
        self.calibrator = None
        self.detector = None
        self.tracker = None
        self.flow_estimator = None
        
        # System state
        self.is_initialized = False
        self.is_calibrated = False
        self.frame_count = 0
        self.calibration_frame_count = 0
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0.0
        
        # Speed history for smoothing
        self.speed_history = defaultdict(list)
        self.speed_smoothing_window = 5
        
        # Previous frame for optical flow
        self.previous_frame = None
        
        print(f"Initialized VehicleSpeedEstimator:")
        print(f"  Camera height: {camera_height}m")
        print(f"  Confidence threshold: {confidence_threshold}")
        print(f"  Optical flow: {flow_method if use_optical_flow else 'Disabled'}")
    
    def initialize_components(self, frame_width: int, frame_height: int, fps: int = 30):
        """
        Initialize system components with video parameters
        
        Args:
            frame_width: Video frame width
            frame_height: Video frame height
            fps: Video frame rate
        """
        print(f"Initializing components for {frame_width}x{frame_height} @ {fps}fps")
        
        # Initialize calibrator
        self.calibrator = CameraCalibrator(
            image_width=frame_width,
            image_height=frame_height,
            assumed_camera_height=self.camera_height
        )
        
        # Initialize detector
        self.detector = VehicleDetector(
            confidence_threshold=self.confidence_threshold
        )
        
        # Initialize tracker
        self.tracker = VehicleTracker(
            track_thresh=self.confidence_threshold,
            frame_rate=fps
        )
        
        # Initialize optical flow estimator
        if self.use_optical_flow:
            self.flow_estimator = OpticalFlowEstimator(
                method='farneback'  # Most reliable for real-time
            )
        
        self.is_initialized = True
        print("All components initialized successfully")
    
    def process_frame(self, frame: np.ndarray) -> Dict:
        """
        Process a single frame and estimate vehicle speeds
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Dictionary containing processing results
        """
        if not self.is_initialized:
            h, w = frame.shape[:2]
            self.initialize_components(w, h)
        
        self.frame_count += 1
        start_time = time.time()
        
        # Initialize result dictionary
        result = {
            'frame_id': self.frame_count,
            'timestamp': start_time,
            'detections': [],
            'tracked_vehicles': [],
            'speeds': {},
            'calibration_status': self.is_calibrated,
            'processing_time': 0.0,
            'fps': 0.0
        }
        
        try:
            # Step 1: Detect vehicles
            detections = self.detector.detect_and_filter(frame)
            result['detections'] = detections
            
            # Step 2: Perform calibration if needed
            if not self.is_calibrated and self.calibration_frame_count < self.calibration_frames:
                self.calibration_frame_count += 1
                
                if len(detections) > 0:  # Only calibrate when vehicles are present
                    success = self.calibrator.calibrate_from_frame(frame, detections)
                    if success:
                        self.is_calibrated = True
                        print(f"Camera calibrated successfully at frame {self.frame_count}")
                        result['calibration_status'] = True
            
            # Step 3: Track vehicles
            tracked_vehicles = self.tracker.update(detections)
            result['tracked_vehicles'] = tracked_vehicles
            
            # Step 4: Calculate speeds if calibrated
            if self.is_calibrated and len(tracked_vehicles) > 0:
                speeds = self.calculate_vehicle_speeds(tracked_vehicles)
                result['speeds'] = speeds
            
            # Step 5: Optional optical flow enhancement
            if self.use_optical_flow and self.previous_frame is not None:
                flow_result = self.enhance_with_optical_flow(
                    self.previous_frame, frame, tracked_vehicles
                )
                result['optical_flow'] = flow_result
            
            # Update previous frame
            self.previous_frame = frame.copy()
            
        except Exception as e:
            print(f"Error processing frame {self.frame_count}: {e}")
            result['error'] = str(e)
        
        # Calculate processing time and FPS
        processing_time = time.time() - start_time
        result['processing_time'] = processing_time
        
        # Update FPS counter
        self.fps_counter += 1
        if self.fps_counter % 10 == 0:  # Update FPS every 10 frames
            current_time = time.time()
            elapsed = current_time - self.fps_start_time
            self.current_fps = 10 / elapsed if elapsed > 0 else 0
            self.fps_start_time = current_time
        
        result['fps'] = self.current_fps
        
        return result
    
    def calculate_vehicle_speeds(self, tracked_vehicles: List[Dict]) -> Dict[int, float]:
        """
        Calculate speeds for tracked vehicles
        
        Args:
            tracked_vehicles: List of tracked vehicle data
            
        Returns:
            Dictionary mapping track IDs to speeds in km/h
        """
        speeds = {}
        
        for vehicle in tracked_vehicles:
            track_id = vehicle['track_id']
            
            # Calculate speed using trajectory
            speed = self.tracker.calculate_track_speed(track_id, self.calibrator)
            
            if speed is not None:
                # Apply smoothing
                self.speed_history[track_id].append(speed)
                if len(self.speed_history[track_id]) > self.speed_smoothing_window:
                    self.speed_history[track_id].pop(0)
                
                # Calculate smoothed speed
                smoothed_speed = np.mean(self.speed_history[track_id])
                speeds[track_id] = smoothed_speed
        
        return speeds
    
    def enhance_with_optical_flow(self, prev_frame: np.ndarray, 
                                 curr_frame: np.ndarray,
                                 tracked_vehicles: List[Dict]) -> Dict:
        """
        Enhance speed estimation using optical flow
        
        Args:
            prev_frame: Previous frame
            curr_frame: Current frame
            tracked_vehicles: List of tracked vehicles
            
        Returns:
            Dictionary with optical flow results
        """
        if self.flow_estimator is None:
            return {}
        
        try:
            # Compute dense optical flow
            flow = self.flow_estimator.compute_dense_flow(prev_frame, curr_frame)
            
            # Extract flow for each vehicle
            vehicle_flows = {}
            for vehicle in tracked_vehicles:
                track_id = vehicle['track_id']
                bbox = vehicle['bbox']
                
                flow_stats = self.flow_estimator.extract_vehicle_flow(flow, bbox)
                vehicle_flows[track_id] = flow_stats
            
            return {
                'flow_field': flow,
                'vehicle_flows': vehicle_flows
            }
            
        except Exception as e:
            print(f"Error computing optical flow: {e}")
            return {}
    
    def visualize_results(self, frame: np.ndarray, result: Dict,
                         show_detections: bool = True,
                         show_tracking: bool = True,
                         show_speeds: bool = True,
                         show_calibration: bool = True,
                         show_flow: bool = False) -> np.ndarray:
        """
        Create comprehensive visualization of the speed estimation results
        
        Args:
            frame: Input frame
            result: Processing result from process_frame()
            show_detections: Show vehicle detections
            show_tracking: Show tracking trajectories
            show_speeds: Show speed estimates
            show_calibration: Show calibration info
            show_flow: Show optical flow visualization
            
        Returns:
            Visualization image
        """
        vis_frame = frame.copy()
        
        # Show calibration overlay
        if show_calibration and self.is_calibrated:
            vis_frame = self.calibrator.visualize_calibration(vis_frame)
        
        # Show detections
        if show_detections and 'detections' in result:
            vis_frame = self.detector.visualize_detections(
                vis_frame, result['detections'], 
                show_confidence=True, show_class=True
            )
        
        # Show tracking and speeds
        if show_tracking and 'tracked_vehicles' in result:
            speeds = result.get('speeds', {})
            vis_frame = self.tracker.visualize_tracks(
                vis_frame, result['tracked_vehicles'],
                show_trajectories=True, show_speeds=show_speeds,
                speeds=speeds
            )
        
        # Show optical flow
        if show_flow and 'optical_flow' in result and 'flow_field' in result['optical_flow']:
            flow_overlay = self.flow_estimator.visualize_vehicle_flow(
                frame, result['tracked_vehicles'], result['optical_flow']['flow_field']
            )
            # Blend with main visualization
            vis_frame = cv2.addWeighted(vis_frame, 0.7, flow_overlay, 0.3, 0)
        
        # Add system status overlay
        self.draw_system_status(vis_frame, result)
        
        return vis_frame
    
    def draw_system_status(self, image: np.ndarray, result: Dict):
        """Draw system status information on the image"""
        h, w = image.shape[:2]
        
        # Status information
        status_lines = [
            f"Frame: {result['frame_id']}",
            f"FPS: {result['fps']:.1f}",
            f"Processing: {result['processing_time']*1000:.1f}ms",
            f"Calibrated: {'Yes' if result['calibration_status'] else 'No'}",
            f"Vehicles: {len(result.get('tracked_vehicles', []))}",
            f"Speeds: {len(result.get('speeds', {}))}"
        ]
        
        # Draw background rectangle
        overlay = image.copy()
        cv2.rectangle(overlay, (w - 250, 10), (w - 10, 10 + len(status_lines) * 25 + 10), 
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
        
        # Draw status text
        for i, line in enumerate(status_lines):
            y_pos = 30 + i * 25
            cv2.putText(image, line, (w - 240, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Draw speed alerts for high-speed vehicles
        speeds = result.get('speeds', {})
        for track_id, speed in speeds.items():
            if speed > 50:  # Alert for speeds > 50 km/h
                cv2.putText(image, f"HIGH SPEED ALERT: {speed:.1f} km/h",
                           (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                break
    
    def get_system_stats(self) -> Dict:
        """Get comprehensive system statistics"""
        stats = {
            'frame_count': self.frame_count,
            'is_calibrated': self.is_calibrated,
            'current_fps': self.current_fps,
            'calibration_frames_used': self.calibration_frame_count
        }
        
        if self.detector:
            stats['detection_stats'] = self.detector.get_detection_stats()
        
        if self.calibrator:
            stats['calibration_info'] = self.calibrator.get_calibration_info()
        
        if self.flow_estimator:
            stats['flow_stats'] = self.flow_estimator.get_flow_statistics()
        
        return stats
    
    def save_calibration(self, filepath: str):
        """Save calibration parameters to file"""
        if not self.is_calibrated:
            print("Warning: System not calibrated, cannot save calibration")
            return
        
        calibration_data = self.calibrator.get_calibration_info()
        calibration_data['camera_height'] = self.camera_height
        calibration_data['system_params'] = {
            'confidence_threshold': self.confidence_threshold,
            'use_optical_flow': self.use_optical_flow
        }
        
        with open(filepath, 'w') as f:
            json.dump(calibration_data, f, indent=2)
        
        print(f"Calibration saved to {filepath}")
    
    def load_calibration(self, filepath: str) -> bool:
        """Load calibration parameters from file"""
        try:
            with open(filepath, 'r') as f:
                calibration_data = json.load(f)
            
            if self.calibrator:
                # Restore calibration parameters
                self.calibrator.focal_length = calibration_data.get('focal_length')
                self.calibrator.principal_point = calibration_data.get('principal_point')
                self.calibrator.scale_factor = calibration_data.get('scale_factor')
                
                if calibration_data.get('homography_matrix'):
                    self.calibrator.homography_matrix = np.array(calibration_data['homography_matrix'])
                
                if calibration_data.get('rotation_matrix'):
                    self.calibrator.rotation_matrix = np.array(calibration_data['rotation_matrix'])
                
                self.calibrator.is_calibrated = calibration_data.get('is_calibrated', False)
                self.is_calibrated = self.calibrator.is_calibrated
                
                print(f"Calibration loaded from {filepath}")
                return True
            
        except Exception as e:
            print(f"Error loading calibration: {e}")
            return False
    
    def cleanup(self):
        """Cleanup resources and perform final statistics"""
        if self.tracker:
            self.tracker.cleanup_old_tracks()
        
        print(f"\nSystem Statistics:")
        print(f"  Total frames processed: {self.frame_count}")
        print(f"  Average FPS: {self.current_fps:.1f}")
        print(f"  Calibration successful: {self.is_calibrated}")
        
        if self.detector:
            det_stats = self.detector.get_detection_stats()
            print(f"  Total detections: {det_stats.get('total_detections', 0)}")
            print(f"  Average detection time: {det_stats.get('average_inference_time', 0)*1000:.1f}ms")