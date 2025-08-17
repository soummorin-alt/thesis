"""
Vanishing Point Detection for Camera Calibration
Implementation based on Dubská & Herout (BMVA 2014)
Automatic camera calibration via vanishing points for vehicle speed estimation
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
import math
from sklearn.cluster import DBSCAN


class VanishingPointDetector:
    """
    Automatic vanishing point detection for camera self-calibration
    Based on Dubská & Herout method using line segment detection and clustering
    """
    
    def __init__(self, 
                 min_line_length: int = 50,
                 max_line_gap: int = 10,
                 rho: float = 1.0,
                 theta: float = np.pi/180,
                 threshold: int = 50):
        """
        Initialize vanishing point detector
        
        Args:
            min_line_length: Minimum length of line segments
            max_line_gap: Maximum gap between line segments
            rho: Distance resolution for Hough transform
            theta: Angle resolution for Hough transform
            threshold: Accumulator threshold for Hough transform
        """
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap
        self.rho = rho
        self.theta = theta
        self.threshold = threshold
        
    def detect_line_segments(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect line segments using Hough Line Transform
        
        Args:
            image: Input grayscale image
            
        Returns:
            List of line segments as (x1, y1, x2, y2) tuples
        """
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        
        # Edge detection using Canny
        edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
        
        # Detect lines using HoughLinesP
        lines = cv2.HoughLinesP(
            edges, 
            self.rho, 
            self.theta, 
            self.threshold,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap
        )
        
        if lines is None:
            return []
            
        # Convert to list of tuples
        line_segments = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            line_segments.append((x1, y1, x2, y2))
            
        return line_segments
    
    def line_intersection(self, line1: Tuple[int, int, int, int], 
                         line2: Tuple[int, int, int, int]) -> Optional[Tuple[float, float]]:
        """
        Find intersection point of two line segments
        
        Args:
            line1: First line segment (x1, y1, x2, y2)
            line2: Second line segment (x1, y1, x2, y2)
            
        Returns:
            Intersection point (x, y) or None if lines are parallel
        """
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2
        
        # Calculate line directions
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        
        # Lines are parallel
        if abs(denom) < 1e-6:
            return None
            
        # Calculate intersection point
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        
        return (x, y)
    
    def are_lines_parallel(self, line1: Tuple[int, int, int, int], 
                          line2: Tuple[int, int, int, int], 
                          angle_threshold: float = 10.0) -> bool:
        """
        Check if two lines are approximately parallel
        
        Args:
            line1: First line segment
            line2: Second line segment  
            angle_threshold: Maximum angle difference in degrees
            
        Returns:
            True if lines are approximately parallel
        """
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2
        
        # Calculate angles
        angle1 = math.atan2(y2 - y1, x2 - x1)
        angle2 = math.atan2(y4 - y3, x4 - x3)
        
        # Normalize angles to [0, π]
        angle1 = abs(angle1)
        angle2 = abs(angle2)
        
        # Calculate angle difference
        angle_diff = abs(angle1 - angle2)
        angle_diff = min(angle_diff, math.pi - angle_diff)
        
        return math.degrees(angle_diff) < angle_threshold
    
    def cluster_vanishing_points(self, points: List[Tuple[float, float]], 
                                eps: float = 50.0, 
                                min_samples: int = 3) -> List[Tuple[float, float]]:
        """
        Cluster intersection points to find vanishing points using DBSCAN
        
        Args:
            points: List of intersection points
            eps: Maximum distance between points in same cluster
            min_samples: Minimum samples per cluster
            
        Returns:
            List of vanishing points (cluster centroids)
        """
        if len(points) < min_samples:
            return []
            
        # Convert to numpy array
        points_array = np.array(points)
        
        # Apply DBSCAN clustering
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points_array)
        
        vanishing_points = []
        
        # Calculate centroid of each cluster
        for cluster_id in set(clustering.labels_):
            if cluster_id == -1:  # Noise points
                continue
                
            cluster_points = points_array[clustering.labels_ == cluster_id]
            centroid = np.mean(cluster_points, axis=0)
            vanishing_points.append((centroid[0], centroid[1]))
            
        return vanishing_points
    
    def detect_vanishing_points(self, image: np.ndarray, 
                               max_vps: int = 3) -> List[Tuple[float, float]]:
        """
        Detect vanishing points in the image
        
        Args:
            image: Input image (BGR or grayscale)
            max_vps: Maximum number of vanishing points to detect
            
        Returns:
            List of vanishing points as (x, y) coordinates
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # Detect line segments
        line_segments = self.detect_line_segments(gray)
        
        if len(line_segments) < 2:
            return []
        
        # Find intersections of parallel lines
        intersection_points = []
        
        for i in range(len(line_segments)):
            for j in range(i + 1, len(line_segments)):
                line1 = line_segments[i]
                line2 = line_segments[j]
                
                # Only consider approximately parallel lines
                if self.are_lines_parallel(line1, line2, angle_threshold=15.0):
                    intersection = self.line_intersection(line1, line2)
                    
                    if intersection is not None:
                        x, y = intersection
                        
                        # Filter out intersections too close to image borders
                        h, w = gray.shape
                        if (-w < x < 2*w and -h < y < 2*h):
                            intersection_points.append(intersection)
        
        # Cluster intersection points to find vanishing points
        vanishing_points = self.cluster_vanishing_points(intersection_points)
        
        # Return at most max_vps vanishing points
        return vanishing_points[:max_vps]
    
    def estimate_focal_length(self, vp1: Tuple[float, float], 
                             vp2: Tuple[float, float],
                             principal_point: Tuple[float, float]) -> float:
        """
        Estimate focal length from two orthogonal vanishing points
        Based on Dubská & Herout orthogonality constraint
        
        Args:
            vp1: First vanishing point
            vp2: Second vanishing point (should be orthogonal to first)
            principal_point: Principal point (cx, cy)
            
        Returns:
            Estimated focal length
        """
        ux, uy = vp1
        vx, vy = vp2
        cx, cy = principal_point
        
        # Apply orthogonality constraint: (u-c)·(v-c) + f² = 0
        # Therefore: f = sqrt(-((ux-cx)(vx-cx) + (uy-cy)(vy-cy)))
        
        dot_product = (ux - cx) * (vx - cx) + (uy - cy) * (vy - cy)
        
        if dot_product >= 0:
            # Vanishing points are not orthogonal
            return None
            
        focal_length = math.sqrt(-dot_product)
        return focal_length
    
    def visualize_vanishing_points(self, image: np.ndarray, 
                                  vanishing_points: List[Tuple[float, float]],
                                  line_segments: List[Tuple[int, int, int, int]] = None) -> np.ndarray:
        """
        Visualize detected vanishing points and line segments
        
        Args:
            image: Input image
            vanishing_points: List of detected vanishing points
            line_segments: Optional list of line segments to draw
            
        Returns:
            Visualization image
        """
        vis_image = image.copy()
        
        # Draw line segments if provided
        if line_segments:
            for x1, y1, x2, y2 in line_segments:
                cv2.line(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 1)
        
        # Draw vanishing points
        colors = [(0, 0, 255), (255, 0, 0), (0, 255, 255)]  # Red, Blue, Yellow
        
        for i, (x, y) in enumerate(vanishing_points):
            color = colors[i % len(colors)]
            
            # Draw vanishing point
            cv2.circle(vis_image, (int(x), int(y)), 8, color, -1)
            cv2.circle(vis_image, (int(x), int(y)), 12, color, 2)
            
            # Label vanishing point
            cv2.putText(vis_image, f'VP{i+1}', 
                       (int(x) + 15, int(y) - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return vis_image