"""
ByteTrack Integration for Vehicle Tracking
Multi-object tracking optimized for vehicle speed estimation
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import math

# ByteTrack imports (assuming ByteTrack is installed)
try:
    from yolox.tracker.byte_tracker import BYTETracker
    from yolox.tracker.basetrack import TrackState
    BYTETRACK_AVAILABLE = True
except ImportError:
    print("Warning: ByteTrack not available, using fallback tracker")
    BYTETRACK_AVAILABLE = False


class VehicleTracker:
    """
    Vehicle tracking using ByteTrack algorithm
    Maintains vehicle trajectories for speed estimation
    """
    
    def __init__(self, 
                 track_thresh: float = 0.5,
                 track_buffer: int = 30,
                 match_thresh: float = 0.8,
                 frame_rate: int = 30):
        """
        Initialize vehicle tracker
        
        Args:
            track_thresh: Detection confidence threshold for tracking
            track_buffer: Number of frames to keep lost tracks
            match_thresh: Matching threshold for track association
            frame_rate: Video frame rate
        """
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.frame_rate = frame_rate
        
        # Initialize ByteTracker if available
        if BYTETRACK_AVAILABLE:
            self.tracker = BYTETracker(
                track_thresh=track_thresh,
                track_buffer=track_buffer,
                match_thresh=match_thresh,
                frame_rate=frame_rate
            )
        else:
            self.tracker = SimpleTracker()
        
        # Track history for speed calculation
        self.track_history = defaultdict(list)
        self.track_speeds = defaultdict(list)
        
        # Frame counter
        self.frame_count = 0
        
    def update(self, detections: List[Dict]) -> List[Dict]:
        """
        Update tracker with new detections
        
        Args:
            detections: List of vehicle detections from current frame
            
        Returns:
            List of tracked vehicles with track IDs
        """
        self.frame_count += 1
        
        if len(detections) == 0:
            if BYTETRACK_AVAILABLE:
                online_targets = self.tracker.update(np.empty((0, 5)), None, None)
            else:
                online_targets = self.tracker.update([])
            return []
        
        # Convert detections to ByteTracker format
        if BYTETRACK_AVAILABLE:
            # Format: [x1, y1, x2, y2, confidence]
            detection_array = np.array([
                [det['bbox'][0], det['bbox'][1], det['bbox'][2], det['bbox'][3], det['confidence']]
                for det in detections
            ])
            
            # Update tracker
            online_targets = self.tracker.update(detection_array, None, None)
            
            # Convert back to our format
            tracked_vehicles = []
            for track in online_targets:
                if track.state == TrackState.Tracked:
                    x1, y1, x2, y2 = track.tlbr
                    
                    tracked_vehicle = {
                        'track_id': track.track_id,
                        'bbox': (int(x1), int(y1), int(x2), int(y2)),
                        'center': (int((x1 + x2) / 2), int((y1 + y2) / 2)),
                        'confidence': track.score,
                        'frame_id': self.frame_count,
                        'age': track.frame_id - track.start_frame + 1
                    }
                    
                    # Find matching detection to get class info
                    for det in detections:
                        det_center = det['center']
                        track_center = tracked_vehicle['center']
                        distance = math.sqrt((det_center[0] - track_center[0])**2 + 
                                           (det_center[1] - track_center[1])**2)
                        if distance < 50:  # Reasonable matching threshold
                            tracked_vehicle.update({
                                'class_id': det['class_id'],
                                'class_name': det['class_name']
                            })
                            break
                    
                    tracked_vehicles.append(tracked_vehicle)
                    
                    # Update track history
                    self.track_history[track.track_id].append({
                        'frame': self.frame_count,
                        'center': tracked_vehicle['center'],
                        'bbox': tracked_vehicle['bbox'],
                        'timestamp': self.frame_count / self.frame_rate
                    })
        else:
            # Use simple tracker fallback
            tracked_vehicles = self.tracker.update(detections)
            
            # Update track history for simple tracker
            for vehicle in tracked_vehicles:
                track_id = vehicle['track_id']
                self.track_history[track_id].append({
                    'frame': self.frame_count,
                    'center': vehicle['center'],
                    'bbox': vehicle['bbox'],
                    'timestamp': self.frame_count / self.frame_rate
                })
        
        return tracked_vehicles
    
    def get_track_trajectory(self, track_id: int, max_length: int = 10) -> List[Dict]:
        """
        Get recent trajectory points for a track
        
        Args:
            track_id: Track ID
            max_length: Maximum number of trajectory points to return
            
        Returns:
            List of trajectory points with positions and timestamps
        """
        if track_id not in self.track_history:
            return []
        
        trajectory = self.track_history[track_id]
        return trajectory[-max_length:] if len(trajectory) > max_length else trajectory
    
    def calculate_track_speed(self, track_id: int, calibrator, 
                            min_trajectory_length: int = 5) -> Optional[float]:
        """
        Calculate speed for a specific track using camera calibration
        
        Args:
            track_id: Track ID
            calibrator: Camera calibrator for ground plane projection
            min_trajectory_length: Minimum trajectory points needed
            
        Returns:
            Speed in km/h or None if insufficient data
        """
        trajectory = self.get_track_trajectory(track_id)
        
        if len(trajectory) < min_trajectory_length:
            return None
        
        if not calibrator.is_calibrated:
            return None
        
        try:
            # Use first and last points for speed calculation
            start_point = trajectory[0]
            end_point = trajectory[-1]
            
            # Calculate ground plane distance
            start_image_point = start_point['center']
            end_image_point = end_point['center']
            
            distance_meters = calibrator.calculate_ground_distance(
                start_image_point, end_image_point
            )
            
            # Calculate time difference
            time_diff = end_point['timestamp'] - start_point['timestamp']
            
            if time_diff <= 0:
                return None
            
            # Calculate speed (m/s to km/h)
            speed_ms = distance_meters / time_diff
            speed_kmh = speed_ms * 3.6
            
            # Store speed for averaging
            self.track_speeds[track_id].append(speed_kmh)
            
            # Return smoothed speed (average of recent measurements)
            recent_speeds = self.track_speeds[track_id][-5:]  # Last 5 measurements
            return np.mean(recent_speeds)
            
        except Exception as e:
            print(f"Error calculating speed for track {track_id}: {e}")
            return None
    
    def get_all_track_speeds(self, calibrator) -> Dict[int, float]:
        """
        Get current speeds for all active tracks
        
        Args:
            calibrator: Camera calibrator
            
        Returns:
            Dictionary mapping track IDs to speeds in km/h
        """
        speeds = {}
        for track_id in self.track_history:
            speed = self.calculate_track_speed(track_id, calibrator)
            if speed is not None:
                speeds[track_id] = speed
        return speeds
    
    def visualize_tracks(self, image: np.ndarray, 
                        tracked_vehicles: List[Dict],
                        show_trajectories: bool = True,
                        show_speeds: bool = True,
                        speeds: Dict[int, float] = None) -> np.ndarray:
        """
        Visualize tracked vehicles with trajectories and speeds
        
        Args:
            image: Input image
            tracked_vehicles: List of tracked vehicles
            show_trajectories: Whether to show trajectory lines
            show_speeds: Whether to show speed information
            speeds: Dictionary of track speeds
            
        Returns:
            Visualization image
        """
        vis_image = image.copy()
        
        # Color palette for tracks
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0)
        ]
        
        for vehicle in tracked_vehicles:
            track_id = vehicle['track_id']
            bbox = vehicle['bbox']
            center = vehicle['center']
            
            # Get color for this track
            color = colors[track_id % len(colors)]
            
            # Draw bounding box
            x1, y1, x2, y2 = bbox
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
            
            # Draw track ID
            cv2.putText(vis_image, f"ID: {track_id}",
                       (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Draw trajectory
            if show_trajectories:
                trajectory = self.get_track_trajectory(track_id, max_length=20)
                if len(trajectory) > 1:
                    points = [point['center'] for point in trajectory]
                    for i in range(1, len(points)):
                        cv2.line(vis_image, points[i-1], points[i], color, 2)
            
            # Draw speed
            if show_speeds and speeds and track_id in speeds:
                speed = speeds[track_id]
                speed_text = f"{speed:.1f} km/h"
                cv2.putText(vis_image, speed_text,
                           (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return vis_image
    
    def cleanup_old_tracks(self, max_age: int = 100):
        """
        Remove old track history to prevent memory buildup
        
        Args:
            max_age: Maximum age of track history to keep (in frames)
        """
        current_frame = self.frame_count
        tracks_to_remove = []
        
        for track_id, history in self.track_history.items():
            if history:
                last_frame = history[-1]['frame']
                if current_frame - last_frame > max_age:
                    tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.track_history[track_id]
            if track_id in self.track_speeds:
                del self.track_speeds[track_id]


class SimpleTracker:
    """
    Simple fallback tracker when ByteTrack is not available
    Uses IoU-based matching for basic tracking
    """
    
    def __init__(self, max_disappeared: int = 10, max_distance: float = 100):
        self.next_id = 0
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
    
    def update(self, detections: List[Dict]) -> List[Dict]:
        """Simple tracking update using centroid matching"""
        if len(detections) == 0:
            # Mark all existing objects as disappeared
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return []
        
        if len(self.objects) == 0:
            # Register all detections as new objects
            for detection in detections:
                self.register(detection)
        else:
            # Match detections to existing objects
            object_centroids = np.array([obj['center'] for obj in self.objects.values()])
            detection_centroids = np.array([det['center'] for det in detections])
            
            # Compute distance matrix
            D = np.linalg.norm(object_centroids[:, np.newaxis] - detection_centroids, axis=2)
            
            # Find minimum distance assignments
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            
            used_row_indices = set()
            used_col_indices = set()
            
            for (row, col) in zip(rows, cols):
                if row in used_row_indices or col in used_col_indices:
                    continue
                
                if D[row, col] > self.max_distance:
                    continue
                
                object_id = list(self.objects.keys())[row]
                self.objects[object_id] = detections[col]
                self.objects[object_id]['track_id'] = object_id
                self.disappeared[object_id] = 0
                
                used_row_indices.add(row)
                used_col_indices.add(col)
            
            # Handle unmatched detections and objects
            unused_row_indices = set(range(0, D.shape[0])).difference(used_row_indices)
            unused_col_indices = set(range(0, D.shape[1])).difference(used_col_indices)
            
            if D.shape[0] >= D.shape[1]:
                for row in unused_row_indices:
                    object_id = list(self.objects.keys())[row]
                    self.disappeared[object_id] += 1
                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)
            else:
                for col in unused_col_indices:
                    self.register(detections[col])
        
        # Return tracked objects
        tracked = []
        for obj in self.objects.values():
            obj_copy = obj.copy()
            tracked.append(obj_copy)
        
        return tracked
    
    def register(self, detection: Dict):
        """Register a new object"""
        detection['track_id'] = self.next_id
        self.objects[self.next_id] = detection
        self.disappeared[self.next_id] = 0
        self.next_id += 1
    
    def deregister(self, object_id: int):
        """Deregister an object"""
        del self.objects[object_id]
        del self.disappeared[object_id]