#!/usr/bin/env python3
"""
Vehicle Speed Estimation from Video File
Process pre-recorded videos and optionally save results
"""

import cv2
import argparse
import sys
import os
import time
import json
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.speed_estimator import VehicleSpeedEstimator


def main():
    parser = argparse.ArgumentParser(description='Vehicle speed estimation from video file')
    parser.add_argument('input_video', help='Input video file path')
    parser.add_argument('--camera_height', type=float, default=1.7,
                       help='Camera height above ground in meters (default: 1.7)')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='Detection confidence threshold (default: 0.5)')
    parser.add_argument('--no_optical_flow', action='store_true',
                       help='Disable optical flow enhancement')
    parser.add_argument('--output_video', type=str, help='Save output video to file')
    parser.add_argument('--load_calibration', type=str, help='Load calibration from file')
    parser.add_argument('--save_calibration', type=str, help='Save calibration to file')
    parser.add_argument('--save_results', type=str, help='Save speed results to JSON file')
    parser.add_argument('--display', action='store_true', help='Display video during processing')
    parser.add_argument('--start_frame', type=int, default=0, help='Start processing from frame N')
    parser.add_argument('--max_frames', type=int, help='Maximum number of frames to process')
    parser.add_argument('--skip_frames', type=int, default=1, help='Process every Nth frame (default: 1)')
    
    args = parser.parse_args()
    
    # Check input file exists
    if not os.path.exists(args.input_video):
        print(f"Error: Input video file not found: {args.input_video}")
        return 1
    
    # Initialize speed estimator
    print("Initializing Vehicle Speed Estimation System...")
    estimator = VehicleSpeedEstimator(
        camera_height=args.camera_height,
        confidence_threshold=args.confidence,
        use_optical_flow=not args.no_optical_flow,
        calibration_frames=100  # Use more frames for video processing
    )
    
    # Load calibration if specified
    if args.load_calibration and os.path.exists(args.load_calibration):
        estimator.load_calibration(args.load_calibration)
        print(f"Loaded calibration from: {args.load_calibration}")
    
    # Open input video
    print(f"Opening video: {args.input_video}")
    cap = cv2.VideoCapture(args.input_video)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file: {args.input_video}")
        return 1
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"Video properties:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Total frames: {total_frames}")
    print(f"  Duration: {duration:.1f}s")
    
    # Set start frame if specified
    if args.start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
        print(f"Starting from frame: {args.start_frame}")
    
    # Calculate processing range
    max_frames = args.max_frames if args.max_frames else (total_frames - args.start_frame)
    frames_to_process = min(max_frames, total_frames - args.start_frame)
    
    print(f"Will process {frames_to_process} frames (every {args.skip_frames} frame(s))")
    
    # Initialize output video writer
    video_writer = None
    if args.output_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_fps = fps / args.skip_frames  # Adjust FPS for frame skipping
        video_writer = cv2.VideoWriter(args.output_video, fourcc, output_fps, (width, height))
        print(f"Saving output video to: {args.output_video}")
    
    # Initialize results storage
    all_results = []
    speed_data = {}
    
    # Processing loop
    frame_count = 0
    processed_count = 0
    start_time = time.time()
    
    print("\nStarting video processing...")
    if args.display:
        print("Press 'q' to quit, 'p' to pause, space to continue")
    
    try:
        while frame_count < frames_to_process:
            ret, frame = cap.read()
            if not ret:
                print("End of video reached")
                break
            
            frame_count += 1
            
            # Skip frames if specified
            if (frame_count - 1) % args.skip_frames != 0:
                continue
            
            processed_count += 1
            
            # Process frame
            result = estimator.process_frame(frame)
            all_results.append(result)
            
            # Store speed data
            speeds = result.get('speeds', {})
            for track_id, speed in speeds.items():
                if track_id not in speed_data:
                    speed_data[track_id] = []
                speed_data[track_id].append({
                    'frame': frame_count,
                    'timestamp': result['timestamp'],
                    'speed_kmh': speed
                })
            
            # Create visualization
            vis_frame = estimator.visualize_results(
                frame, result,
                show_detections=True,
                show_tracking=True,
                show_speeds=True,
                show_calibration=True,
                show_flow=False
            )
            
            # Add progress information
            progress = (processed_count / (frames_to_process // args.skip_frames)) * 100
            cv2.putText(vis_frame, f"Progress: {progress:.1f}% (Frame {frame_count}/{total_frames})",
                       (10, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Save frame to output video
            if video_writer:
                video_writer.write(vis_frame)
            
            # Display frame if requested
            if args.display:
                cv2.imshow('Vehicle Speed Estimation - Video Processing', vis_frame)
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("Quit requested by user")
                    break
                elif key == ord('p'):
                    print("Paused - press space to continue")
                    while True:
                        key = cv2.waitKey(0) & 0xFF
                        if key == ord(' '):
                            break
                        elif key == ord('q'):
                            break
                    if key == ord('q'):
                        break
            
            # Print progress every 100 frames
            if processed_count % 100 == 0:
                elapsed = time.time() - start_time
                fps_current = processed_count / elapsed if elapsed > 0 else 0
                eta = (frames_to_process // args.skip_frames - processed_count) / fps_current if fps_current > 0 else 0
                
                print(f"Processed {processed_count} frames ({progress:.1f}%), "
                      f"FPS: {fps_current:.1f}, ETA: {eta:.1f}s")
                
                # Print calibration status
                if processed_count == 100:
                    print(f"Calibration status: {estimator.is_calibrated}")
                
                # Print current speeds
                if speeds:
                    speed_str = ", ".join([f"ID{tid}:{speed:.1f}km/h" 
                                         for tid, speed in speeds.items()])
                    print(f"Current speeds: {speed_str}")
    
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
    
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        print("\nProcessing complete. Cleaning up...")
        
        # Calculate final statistics
        total_time = time.time() - start_time
        avg_fps = processed_count / total_time if total_time > 0 else 0
        
        print(f"\nProcessing Summary:")
        print(f"  Frames processed: {processed_count}")
        print(f"  Total time: {total_time:.1f}s")
        print(f"  Average FPS: {avg_fps:.1f}")
        print(f"  Calibration successful: {estimator.is_calibrated}")
        
        # Print speed statistics
        if speed_data:
            print(f"\nSpeed Statistics:")
            for track_id, speeds in speed_data.items():
                if speeds:
                    speed_values = [s['speed_kmh'] for s in speeds]
                    avg_speed = sum(speed_values) / len(speed_values)
                    max_speed = max(speed_values)
                    print(f"  Track {track_id}: Avg={avg_speed:.1f}km/h, Max={max_speed:.1f}km/h, Samples={len(speeds)}")
        
        # Save calibration if specified
        if args.save_calibration and estimator.is_calibrated:
            estimator.save_calibration(args.save_calibration)
        
        # Save results to JSON if specified
        if args.save_results:
            results_data = {
                'video_info': {
                    'input_file': args.input_video,
                    'width': width,
                    'height': height,
                    'fps': fps,
                    'total_frames': total_frames,
                    'processed_frames': processed_count
                },
                'processing_params': {
                    'camera_height': args.camera_height,
                    'confidence_threshold': args.confidence,
                    'use_optical_flow': not args.no_optical_flow,
                    'skip_frames': args.skip_frames
                },
                'calibration_info': estimator.get_system_stats(),
                'speed_data': speed_data,
                'frame_results': [
                    {
                        'frame_id': r['frame_id'],
                        'timestamp': r['timestamp'],
                        'speeds': r.get('speeds', {}),
                        'vehicle_count': len(r.get('tracked_vehicles', []))
                    }
                    for r in all_results
                ]
            }
            
            with open(args.save_results, 'w') as f:
                json.dump(results_data, f, indent=2)
            
            print(f"Results saved to: {args.save_results}")
        
        # Release resources
        cap.release()
        if video_writer:
            video_writer.release()
        if args.display:
            cv2.destroyAllWindows()
        
        # Final cleanup
        estimator.cleanup()


if __name__ == '__main__':
    sys.exit(main())