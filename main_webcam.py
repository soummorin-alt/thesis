#!/usr/bin/env python3
"""
Real-Time Vehicle Speed Estimation from Webcam
Main application for processing live webcam feed
"""

import cv2
import argparse
import sys
import os
import time
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.speed_estimator import VehicleSpeedEstimator


def main():
    parser = argparse.ArgumentParser(description='Real-time vehicle speed estimation from webcam')
    parser.add_argument('--camera', type=int, default=0, help='Camera device ID (default: 0)')
    parser.add_argument('--camera_height', type=float, default=1.7, 
                       help='Camera height above ground in meters (default: 1.7)')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='Detection confidence threshold (default: 0.5)')
    parser.add_argument('--no_optical_flow', action='store_true',
                       help='Disable optical flow enhancement')
    parser.add_argument('--save_video', type=str, help='Save output video to file')
    parser.add_argument('--load_calibration', type=str, help='Load calibration from file')
    parser.add_argument('--save_calibration', type=str, help='Save calibration to file')
    parser.add_argument('--width', type=int, default=1280, help='Camera width (default: 1280)')
    parser.add_argument('--height', type=int, default=720, help='Camera height (default: 720)')
    parser.add_argument('--fps', type=int, default=30, help='Camera FPS (default: 30)')
    
    args = parser.parse_args()
    
    # Initialize speed estimator
    print("Initializing Vehicle Speed Estimation System...")
    estimator = VehicleSpeedEstimator(
        camera_height=args.camera_height,
        confidence_threshold=args.confidence,
        use_optical_flow=not args.no_optical_flow,
        calibration_frames=50  # Use more frames for better calibration
    )
    
    # Load calibration if specified
    if args.load_calibration and os.path.exists(args.load_calibration):
        estimator.load_calibration(args.load_calibration)
    
    # Initialize camera
    print(f"Opening camera {args.camera}...")
    cap = cv2.VideoCapture(args.camera)
    
    if not cap.isOpened():
        print(f"Error: Could not open camera {args.camera}")
        return 1
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    
    # Get actual camera properties
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Camera initialized: {actual_width}x{actual_height} @ {actual_fps}fps")
    
    # Initialize video writer if saving
    video_writer = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            args.save_video, fourcc, actual_fps, (actual_width, actual_height)
        )
        print(f"Saving video to: {args.save_video}")
    
    print("\nStarting real-time processing...")
    print("Controls:")
    print("  'q' - Quit")
    print("  'c' - Force recalibration")
    print("  's' - Save calibration")
    print("  'r' - Reset statistics")
    print("  'h' - Toggle help overlay")
    print("\nPress any key to start...")
    
    # Main processing loop
    frame_count = 0
    start_time = time.time()
    show_help = False
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame from camera")
                break
            
            frame_count += 1
            
            # Process frame
            result = estimator.process_frame(frame)
            
            # Create visualization
            vis_frame = estimator.visualize_results(
                frame, result,
                show_detections=True,
                show_tracking=True,
                show_speeds=True,
                show_calibration=True,
                show_flow=False  # Optical flow visualization can be heavy
            )
            
            # Add help overlay if requested
            if show_help:
                draw_help_overlay(vis_frame)
            
            # Display frame
            cv2.imshow('Vehicle Speed Estimation', vis_frame)
            
            # Save frame if recording
            if video_writer:
                video_writer.write(vis_frame)
            
            # Print periodic status
            if frame_count % 100 == 0:
                elapsed = time.time() - start_time
                avg_fps = frame_count / elapsed if elapsed > 0 else 0
                print(f"Frame {frame_count}, Avg FPS: {avg_fps:.1f}, "
                      f"Calibrated: {estimator.is_calibrated}")
                
                # Print current speeds
                speeds = result.get('speeds', {})
                if speeds:
                    speed_str = ", ".join([f"ID{tid}:{speed:.1f}km/h" 
                                         for tid, speed in speeds.items()])
                    print(f"Current speeds: {speed_str}")
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('c'):
                # Force recalibration
                estimator.is_calibrated = False
                estimator.calibration_frame_count = 0
                if estimator.calibrator:
                    estimator.calibrator.is_calibrated = False
                print("Forced recalibration")
            elif key == ord('s'):
                # Save calibration
                if args.save_calibration:
                    estimator.save_calibration(args.save_calibration)
                else:
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"calibration_{timestamp}.json"
                    estimator.save_calibration(filename)
            elif key == ord('r'):
                # Reset statistics
                if estimator.detector:
                    estimator.detector.reset_stats()
                print("Statistics reset")
            elif key == ord('h'):
                # Toggle help
                show_help = not show_help
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        print("\nCleaning up...")
        
        # Print final statistics
        estimator.cleanup()
        
        # Save final calibration if specified
        if args.save_calibration and estimator.is_calibrated:
            estimator.save_calibration(args.save_calibration)
        
        # Release resources
        cap.release()
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()
        
        # Print summary
        elapsed = time.time() - start_time
        avg_fps = frame_count / elapsed if elapsed > 0 else 0
        print(f"\nSession Summary:")
        print(f"  Total frames: {frame_count}")
        print(f"  Duration: {elapsed:.1f}s")
        print(f"  Average FPS: {avg_fps:.1f}")
        print(f"  Final calibration status: {estimator.is_calibrated}")


def draw_help_overlay(image):
    """Draw help information overlay"""
    help_lines = [
        "CONTROLS:",
        "  'q' - Quit",
        "  'c' - Force recalibration", 
        "  's' - Save calibration",
        "  'r' - Reset statistics",
        "  'h' - Toggle this help",
        "",
        "STATUS INDICATORS:",
        "  Green boxes - Detected vehicles",
        "  Colored tracks - Vehicle trajectories", 
        "  Speed values - Current speeds",
        "  Calibration info - Top left corner"
    ]
    
    # Draw semi-transparent background
    overlay = image.copy()
    h, w = image.shape[:2]
    cv2.rectangle(overlay, (10, 10), (400, 10 + len(help_lines) * 25 + 10), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.8, image, 0.2, 0, image)
    
    # Draw help text
    for i, line in enumerate(help_lines):
        y_pos = 30 + i * 25
        cv2.putText(image, line, (20, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)


if __name__ == '__main__':
    sys.exit(main())