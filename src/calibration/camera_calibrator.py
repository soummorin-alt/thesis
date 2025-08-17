"""
Camera Calibration Module
Implements automatic camera calibration using vanishing points and scene geometry
Based on Dubská & Herout methodology for vehicle speed estimation
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
import math
from .vanishing_point_detector import VanishingPointDetector


class CameraCalibrator:
    """
    Automatic camera calibration for monocular vehicle speed estimation
    Uses vanishing point detection and scene geometry constraints
    """
    
    def __init__(self, 
                 image_width: int,
                 image_height: int,
                 assumed_camera_height: float = 1.7):  # meters, typical eye level
        """
        Initialize camera calibrator
        
        Args:
            image_width: Image width in pixels
            image_height: Image height in pixels
            assumed_camera_height: Camera height above ground in meters
        """
        self.image_width = image_width
        self.image_height = image_height
        self.camera_height = assumed_camera_height
        
        # Initialize vanishing point detector
        self.vp_detector = VanishingPointDetector()
        
        # Camera parameters (will be estimated)
        self.focal_length = None
        self.principal_point = None
        self.rotation_matrix = None
        self.homography_matrix = None
        self.scale_factor = None
        
        # Calibration status
        self.is_calibrated = False
        
    def estimate_principal_point(self) -> Tuple[float, float]:
        """
        Estimate principal point (typically at image center)
        
        Returns:
            Principal point coordinates (cx, cy)
        """
        cx = self.image_width / 2.0
        cy = self.image_height / 2.0
        return (cx, cy)
    
    def estimate_camera_parameters(self, vanishing_points: List[Tuple[float, float]]) -> Dict:
        """
        Estimate camera intrinsic parameters from vanishing points
        
        Args:
            vanishing_points: List of detected vanishing points
            
        Returns:
            Dictionary containing camera parameters
        """
        if len(vanishing_points) < 2:
            return None
            
        # Estimate principal point (assume image center)
        self.principal_point = self.estimate_principal_point()
        cx, cy = self.principal_point
        
        # Find the best pair of orthogonal vanishing points
        best_focal_length = None
        best_vp_pair = None
        
        for i in range(len(vanishing_points)):
            for j in range(i + 1, len(vanishing_points)):
                vp1 = vanishing_points[i]
                vp2 = vanishing_points[j]
                
                focal_length = self.vp_detector.estimate_focal_length(vp1, vp2, self.principal_point)
                
                if focal_length is not None and focal_length > 0:
                    # Sanity check: focal length should be reasonable
                    if 0.5 * max(self.image_width, self.image_height) < focal_length < 3.0 * max(self.image_width, self.image_height):
                        best_focal_length = focal_length
                        best_vp_pair = (vp1, vp2)
                        break
        
        if best_focal_length is None:
            # Fallback: estimate focal length from image dimensions
            best_focal_length = max(self.image_width, self.image_height)
            print("Warning: Could not estimate focal length from vanishing points, using fallback")
        
        self.focal_length = best_focal_length
        
        # Create camera intrinsic matrix
        K = np.array([
            [self.focal_length, 0, cx],
            [0, self.focal_length, cy],
            [0, 0, 1]
        ], dtype=np.float32)
        
        return {
            'focal_length': self.focal_length,
            'principal_point': self.principal_point,
            'intrinsic_matrix': K,
            'vanishing_points': best_vp_pair
        }
    
    def estimate_ground_plane_homography(self, vanishing_points: List[Tuple[float, float]]) -> np.ndarray:
        """
        Estimate homography matrix mapping image plane to ground plane
        
        Args:
            vanishing_points: List of detected vanishing points
            
        Returns:
            3x3 homography matrix
        """
        if self.focal_length is None or self.principal_point is None:
            raise ValueError("Camera parameters must be estimated first")
        
        cx, cy = self.principal_point
        f = self.focal_length
        h = self.camera_height
        
        # Create camera intrinsic matrix
        K = np.array([
            [f, 0, cx],
            [0, f, cy],
            [0, 0, 1]
        ], dtype=np.float32)
        
        if len(vanishing_points) >= 1:
            # Use first vanishing point to estimate ground plane orientation
            vp1 = vanishing_points[0]
            
            # Convert vanishing point to normalized coordinates
            vp1_norm = np.array([
                (vp1[0] - cx) / f,
                (vp1[1] - cy) / f,
                1.0
            ])
            
            # Estimate rotation matrix (simplified approach)
            # Assume the first vanishing point corresponds to the traffic direction
            # and the ground plane is approximately horizontal
            
            # Calculate pitch angle from vanishing point
            pitch = math.atan2(vp1_norm[1], vp1_norm[2])
            
            # Create rotation matrix (pitch rotation around x-axis)
            cos_p = math.cos(pitch)
            sin_p = math.sin(pitch)
            
            R = np.array([
                [1, 0, 0],
                [0, cos_p, -sin_p],
                [0, sin_p, cos_p]
            ], dtype=np.float32)
            
        else:
            # Fallback: assume camera is looking straight ahead
            R = np.eye(3, dtype=np.float32)
        
        self.rotation_matrix = R
        
        # Create homography for ground plane (Z=0)
        # H = K * [r1, r2, t] where [r1, r2, r3] = R and t = -R * [0, 0, h]
        r1 = R[:, 0]
        r2 = R[:, 1]
        r3 = R[:, 2]
        
        # Translation vector (camera center to ground plane)
        t = -h * r3
        
        # Construct homography matrix
        H_3x3 = np.column_stack([r1, r2, t])
        H = K @ H_3x3
        
        # Normalize homography
        H = H / H[2, 2]
        
        self.homography_matrix = H
        return H
    
    def estimate_scale_factor(self, detected_vehicles: List[Dict]) -> float:
        """
        Estimate metric scale using known vehicle dimensions
        Based on Dubská & Herout approach
        
        Args:
            detected_vehicles: List of vehicle detections with bounding boxes
            
        Returns:
            Scale factor for converting image measurements to meters
        """
        if len(detected_vehicles) == 0:
            return 1.0
        
        # Standard vehicle dimensions (in meters)
        STANDARD_CAR_LENGTH = 4.2
        STANDARD_CAR_WIDTH = 1.8
        STANDARD_CAR_HEIGHT = 1.5
        
        scale_estimates = []
        
        for vehicle in detected_vehicles:
            bbox = vehicle.get('bbox')  # (x1, y1, x2, y2)
            if bbox is None:
                continue
                
            x1, y1, x2, y2 = bbox
            
            # Calculate bounding box dimensions in pixels
            width_px = x2 - x1
            height_px = y2 - y1
            
            # Estimate vehicle dimensions using ground plane projection
            if self.homography_matrix is not None:
                # Project bottom corners to ground plane
                bottom_left = np.array([x1, y2, 1])
                bottom_right = np.array([x2, y2, 1])
                
                # Transform to ground plane coordinates
                H_inv = np.linalg.inv(self.homography_matrix)
                ground_left = H_inv @ bottom_left
                ground_right = H_inv @ bottom_right
                
                # Normalize homogeneous coordinates
                ground_left = ground_left / ground_left[2]
                ground_right = ground_right / ground_right[2]
                
                # Calculate ground plane width
                ground_width = np.linalg.norm(ground_right[:2] - ground_left[:2])
                
                if ground_width > 0:
                    # Estimate scale factor
                    scale_estimate = STANDARD_CAR_WIDTH / ground_width
                    scale_estimates.append(scale_estimate)
        
        if scale_estimates:
            # Use median scale estimate for robustness
            self.scale_factor = np.median(scale_estimates)
        else:
            # Fallback scale factor
            self.scale_factor = 1.0
            
        return self.scale_factor
    
    def calibrate_from_frame(self, image: np.ndarray, 
                            detected_vehicles: List[Dict] = None) -> bool:
        """
        Perform complete camera calibration from a single frame
        
        Args:
            image: Input image for calibration
            detected_vehicles: Optional list of vehicle detections for scale estimation
            
        Returns:
            True if calibration successful, False otherwise
        """
        try:
            # Detect vanishing points
            vanishing_points = self.vp_detector.detect_vanishing_points(image)
            
            if len(vanishing_points) == 0:
                print("Warning: No vanishing points detected")
                return False
            
            # Estimate camera parameters
            camera_params = self.estimate_camera_parameters(vanishing_points)
            if camera_params is None:
                print("Warning: Could not estimate camera parameters")
                return False
            
            # Estimate ground plane homography
            homography = self.estimate_ground_plane_homography(vanishing_points)
            
            # Estimate scale factor if vehicles are provided
            if detected_vehicles:
                scale_factor = self.estimate_scale_factor(detected_vehicles)
            else:
                self.scale_factor = 1.0
            
            self.is_calibrated = True
            
            print(f"Calibration successful:")
            print(f"  Focal length: {self.focal_length:.1f}")
            print(f"  Principal point: ({self.principal_point[0]:.1f}, {self.principal_point[1]:.1f})")
            print(f"  Scale factor: {self.scale_factor:.3f}")
            print(f"  Vanishing points: {len(vanishing_points)}")
            
            return True
            
        except Exception as e:
            print(f"Calibration failed: {e}")
            return False
    
    def image_to_ground_coordinates(self, image_point: Tuple[float, float]) -> Tuple[float, float]:
        """
        Convert image coordinates to ground plane coordinates
        
        Args:
            image_point: Point in image coordinates (x, y)
            
        Returns:
            Point in ground plane coordinates (X, Y) in meters
        """
        if not self.is_calibrated or self.homography_matrix is None:
            raise ValueError("Camera must be calibrated first")
        
        # Convert to homogeneous coordinates
        point_h = np.array([image_point[0], image_point[1], 1.0])
        
        # Apply inverse homography
        H_inv = np.linalg.inv(self.homography_matrix)
        ground_h = H_inv @ point_h
        
        # Normalize and apply scale factor
        ground_h = ground_h / ground_h[2]
        ground_x = ground_h[0] * self.scale_factor
        ground_y = ground_h[1] * self.scale_factor
        
        return (ground_x, ground_y)
    
    def calculate_ground_distance(self, point1: Tuple[float, float], 
                                 point2: Tuple[float, float]) -> float:
        """
        Calculate distance between two image points in ground plane meters
        
        Args:
            point1: First image point (x, y)
            point2: Second image point (x, y)
            
        Returns:
            Distance in meters
        """
        ground1 = self.image_to_ground_coordinates(point1)
        ground2 = self.image_to_ground_coordinates(point2)
        
        distance = math.sqrt((ground2[0] - ground1[0])**2 + (ground2[1] - ground1[1])**2)
        return distance
    
    def get_calibration_info(self) -> Dict:
        """
        Get current calibration parameters
        
        Returns:
            Dictionary containing calibration information
        """
        return {
            'is_calibrated': self.is_calibrated,
            'focal_length': self.focal_length,
            'principal_point': self.principal_point,
            'camera_height': self.camera_height,
            'scale_factor': self.scale_factor,
            'homography_matrix': self.homography_matrix.tolist() if self.homography_matrix is not None else None,
            'rotation_matrix': self.rotation_matrix.tolist() if self.rotation_matrix is not None else None
        }
    
    def visualize_calibration(self, image: np.ndarray) -> np.ndarray:
        """
        Create visualization of calibration results
        
        Args:
            image: Input image
            
        Returns:
            Visualization image with calibration overlay
        """
        vis_image = image.copy()
        
        if not self.is_calibrated:
            cv2.putText(vis_image, "NOT CALIBRATED", (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return vis_image
        
        # Draw principal point
        if self.principal_point:
            cx, cy = self.principal_point
            cv2.circle(vis_image, (int(cx), int(cy)), 5, (255, 255, 0), -1)
            cv2.putText(vis_image, "Principal Point", (int(cx) + 10, int(cy) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Draw horizon line if available
        if self.homography_matrix is not None:
            # Estimate horizon line from homography
            # This is a simplified visualization
            horizon_y = self.image_height // 2  # Approximate
            cv2.line(vis_image, (0, horizon_y), (self.image_width, horizon_y), (0, 255, 255), 2)
            cv2.putText(vis_image, "Horizon", (10, horizon_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Display calibration info
        info_text = [
            f"Focal Length: {self.focal_length:.1f}" if self.focal_length else "Focal Length: N/A",
            f"Scale Factor: {self.scale_factor:.3f}" if self.scale_factor else "Scale Factor: N/A",
            f"Camera Height: {self.camera_height:.1f}m"
        ]
        
        for i, text in enumerate(info_text):
            cv2.putText(vis_image, text, (10, 30 + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(vis_image, text, (10, 30 + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        return vis_image