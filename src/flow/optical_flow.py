"""
Optical Flow Computation for Vehicle Motion Estimation
Supports both traditional (OpenCV) and deep learning approaches (RAFT)
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict, List
import torch
import torch.nn.functional as F


class OpticalFlowEstimator:
    """
    Optical flow estimation for vehicle motion tracking
    Supports multiple algorithms for robustness
    """
    
    def __init__(self, method: str = 'farneback', use_gpu: bool = True):
        """
        Initialize optical flow estimator
        
        Args:
            method: Flow estimation method ('farneback', 'lucas_kanade', 'raft')
            use_gpu: Whether to use GPU acceleration when available
        """
        self.method = method
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = 'cuda' if self.use_gpu else 'cpu'
        
        # Initialize method-specific parameters
        if method == 'farneback':
            self.flow_params = {
                'pyr_scale': 0.5,
                'levels': 3,
                'winsize': 15,
                'iterations': 3,
                'poly_n': 5,
                'poly_sigma': 1.2,
                'flags': 0
            }
        elif method == 'lucas_kanade':
            self.lk_params = {
                'winSize': (15, 15),
                'maxLevel': 2,
                'criteria': (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
            }
            # Feature detection parameters
            self.feature_params = {
                'maxCorners': 100,
                'qualityLevel': 0.3,
                'minDistance': 7,
                'blockSize': 7
            }
        elif method == 'raft':
            self.raft_model = None
            self.load_raft_model()
        
        print(f"Initialized optical flow estimator: {method} on {self.device}")
    
    def load_raft_model(self):
        """Load pre-trained RAFT model for deep optical flow"""
        try:
            # Try to load RAFT from torchvision (if available)
            from torchvision.models.optical_flow import raft_large
            self.raft_model = raft_large(pretrained=True)
            self.raft_model.eval()
            if self.use_gpu:
                self.raft_model = self.raft_model.cuda()
            print("Loaded RAFT model from torchvision")
        except ImportError:
            print("RAFT not available from torchvision, falling back to Farneback")
            self.method = 'farneback'
            self.raft_model = None
    
    def preprocess_for_raft(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """Preprocess images for RAFT model"""
        def transform_image(img):
            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Normalize to [0, 1]
            img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
            return img_tensor.unsqueeze(0)  # Add batch dimension
        
        img1_tensor = transform_image(img1)
        img2_tensor = transform_image(img2)
        
        if self.use_gpu:
            img1_tensor = img1_tensor.cuda()
            img2_tensor = img2_tensor.cuda()
        
        return img1_tensor, img2_tensor
    
    def compute_dense_flow(self, frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
        """
        Compute dense optical flow between two frames
        
        Args:
            frame1: First frame (BGR)
            frame2: Second frame (BGR)
            
        Returns:
            Flow field as numpy array (H, W, 2) with dx, dy components
        """
        # Convert to grayscale for traditional methods
        if self.method in ['farneback', 'lucas_kanade']:
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        if self.method == 'farneback':
            flow = cv2.calcOpticalFlowPyrLK(gray1, gray2, **self.flow_params)
            return flow
            
        elif self.method == 'raft' and self.raft_model is not None:
            with torch.no_grad():
                img1_tensor, img2_tensor = self.preprocess_for_raft(frame1, frame2)
                flow_predictions = self.raft_model(img1_tensor, img2_tensor)
                
                # Get the final flow prediction
                flow = flow_predictions[-1][0]  # Remove batch dimension
                flow = flow.permute(1, 2, 0).cpu().numpy()  # (H, W, 2)
                
            return flow
        
        else:
            # Fallback to Farneback
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowPyrLK(gray1, gray2, **self.flow_params)
            return flow
    
    def compute_sparse_flow(self, frame1: np.ndarray, frame2: np.ndarray, 
                           points: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute sparse optical flow using Lucas-Kanade
        
        Args:
            frame1: First frame (BGR)
            frame2: Second frame (BGR)
            points: Points to track (Nx1x2), if None will detect features
            
        Returns:
            Tuple of (new_points, status, error)
        """
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        if points is None:
            # Detect good features to track
            points = cv2.goodFeaturesToTrack(gray1, **self.feature_params)
        
        if points is None or len(points) == 0:
            return np.array([]), np.array([]), np.array([])
        
        # Calculate optical flow
        new_points, status, error = cv2.calcOpticalFlowPyrLK(
            gray1, gray2, points, None, **self.lk_params
        )
        
        return new_points, status, error
    
    def extract_vehicle_flow(self, flow: np.ndarray, bbox: Tuple[int, int, int, int]) -> Dict:
        """
        Extract flow statistics within a vehicle bounding box
        
        Args:
            flow: Dense optical flow field (H, W, 2)
            bbox: Bounding box (x1, y1, x2, y2)
            
        Returns:
            Dictionary with flow statistics
        """
        x1, y1, x2, y2 = bbox
        
        # Extract flow within bounding box
        roi_flow = flow[y1:y2, x1:x2]
        
        if roi_flow.size == 0:
            return {
                'mean_flow': (0.0, 0.0),
                'magnitude': 0.0,
                'angle': 0.0,
                'flow_vectors': 0
            }
        
        # Calculate flow statistics
        flow_x = roi_flow[:, :, 0]
        flow_y = roi_flow[:, :, 1]
        
        # Remove outliers (flow vectors with very large magnitude)
        magnitude = np.sqrt(flow_x**2 + flow_y**2)
        valid_mask = magnitude < np.percentile(magnitude, 95)  # Remove top 5% outliers
        
        if np.sum(valid_mask) == 0:
            return {
                'mean_flow': (0.0, 0.0),
                'magnitude': 0.0,
                'angle': 0.0,
                'flow_vectors': 0
            }
        
        # Calculate mean flow
        mean_flow_x = np.mean(flow_x[valid_mask])
        mean_flow_y = np.mean(flow_y[valid_mask])
        
        # Calculate magnitude and angle
        mean_magnitude = np.sqrt(mean_flow_x**2 + mean_flow_y**2)
        mean_angle = np.arctan2(mean_flow_y, mean_flow_x)
        
        return {
            'mean_flow': (float(mean_flow_x), float(mean_flow_y)),
            'magnitude': float(mean_magnitude),
            'angle': float(mean_angle),
            'flow_vectors': int(np.sum(valid_mask))
        }
    
    def track_vehicle_points(self, frame1: np.ndarray, frame2: np.ndarray,
                           vehicle_bbox: Tuple[int, int, int, int]) -> List[Tuple[float, float]]:
        """
        Track specific points within a vehicle using sparse optical flow
        
        Args:
            frame1: First frame
            frame2: Second frame
            vehicle_bbox: Vehicle bounding box
            
        Returns:
            List of displacement vectors for tracked points
        """
        x1, y1, x2, y2 = vehicle_bbox
        
        # Extract region of interest
        roi1 = frame1[y1:y2, x1:x2]
        gray_roi = cv2.cvtColor(roi1, cv2.COLOR_BGR2GRAY)
        
        # Detect features within the vehicle
        points = cv2.goodFeaturesToTrack(gray_roi, **self.feature_params)
        
        if points is None or len(points) == 0:
            return []
        
        # Convert points to global coordinates
        points[:, :, 0] += x1
        points[:, :, 1] += y1
        
        # Track points
        new_points, status, error = self.compute_sparse_flow(frame1, frame2, points)
        
        # Calculate displacements for successful tracks
        displacements = []
        for i, (stat, err) in enumerate(zip(status, error)):
            if stat == 1 and err < 50:  # Good track
                old_point = points[i][0]
                new_point = new_points[i][0]
                displacement = (new_point[0] - old_point[0], new_point[1] - old_point[1])
                displacements.append(displacement)
        
        return displacements
    
    def visualize_flow(self, image: np.ndarray, flow: np.ndarray, 
                      step: int = 16) -> np.ndarray:
        """
        Visualize optical flow as arrows on the image
        
        Args:
            image: Background image
            flow: Optical flow field
            step: Sampling step for visualization
            
        Returns:
            Visualization image
        """
        vis_image = image.copy()
        h, w = flow.shape[:2]
        
        # Create a grid of points
        y, x = np.mgrid[step//2:h:step, step//2:w:step].reshape(2, -1).astype(int)
        
        # Get flow vectors at grid points
        fx, fy = flow[y, x].T
        
        # Create line endpoints
        lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
        lines = np.int32(lines + 0.5)
        
        # Draw flow vectors
        for (x1, y1), (x2, y2) in lines:
            # Skip very small movements
            if abs(x2 - x1) < 1 and abs(y2 - y1) < 1:
                continue
                
            # Draw arrow
            cv2.arrowedLine(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 1, tipLength=0.3)
        
        return vis_image
    
    def visualize_vehicle_flow(self, image: np.ndarray, vehicles: List[Dict],
                             flow: np.ndarray) -> np.ndarray:
        """
        Visualize optical flow for tracked vehicles
        
        Args:
            image: Background image
            vehicles: List of vehicle detections with bounding boxes
            flow: Optical flow field
            
        Returns:
            Visualization image
        """
        vis_image = image.copy()
        
        for vehicle in vehicles:
            bbox = vehicle['bbox']
            x1, y1, x2, y2 = bbox
            
            # Extract flow statistics for this vehicle
            flow_stats = self.extract_vehicle_flow(flow, bbox)
            
            # Draw bounding box
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw flow vector
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            flow_x, flow_y = flow_stats['mean_flow']
            
            # Scale flow for visualization
            scale = 10
            end_x = int(center_x + flow_x * scale)
            end_y = int(center_y + flow_y * scale)
            
            # Draw flow arrow
            cv2.arrowedLine(vis_image, (center_x, center_y), (end_x, end_y), 
                          (0, 0, 255), 3, tipLength=0.3)
            
            # Display flow magnitude
            magnitude = flow_stats['magnitude']
            cv2.putText(vis_image, f"Flow: {magnitude:.1f}px",
                       (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return vis_image
    
    def get_flow_statistics(self) -> Dict:
        """Get performance statistics for the flow estimator"""
        return {
            'method': self.method,
            'device': self.device,
            'gpu_available': torch.cuda.is_available(),
            'raft_loaded': self.raft_model is not None
        }