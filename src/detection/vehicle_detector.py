"""
Vehicle Detection Module
Uses YOLOv8 for real-time vehicle detection in traffic scenes
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple
from ultralytics import YOLO
import torch


class VehicleDetector:
    """
    Real-time vehicle detector using YOLOv8
    Optimized for traffic surveillance and speed estimation
    """
    
    def __init__(self, 
                 model_path: str = 'yolov8n.pt',
                 confidence_threshold: float = 0.5,
                 device: str = 'auto'):
        """
        Initialize vehicle detector
        
        Args:
            model_path: Path to YOLOv8 model weights
            confidence_threshold: Minimum confidence for detections
            device: Device to run inference on ('cpu', 'cuda', 'auto')
        """
        self.confidence_threshold = confidence_threshold
        
        # Auto-detect device
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        print(f"Loading YOLOv8 model on device: {self.device}")
        
        # Load YOLOv8 model
        try:
            self.model = YOLO(model_path)
            self.model.to(self.device)
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
        
        # Vehicle class IDs in COCO dataset
        self.vehicle_classes = {
            2: 'car',
            3: 'motorcycle', 
            5: 'bus',
            7: 'truck'
        }
        
        # Track detection statistics
        self.detection_count = 0
        self.total_inference_time = 0.0
        
    def detect_vehicles(self, image: np.ndarray, 
                       return_crops: bool = False) -> List[Dict]:
        """
        Detect vehicles in the input image
        
        Args:
            image: Input image (BGR format)
            return_crops: Whether to return cropped vehicle images
            
        Returns:
            List of vehicle detections with bounding boxes, confidence, and class
        """
        start_time = cv2.getTickCount()
        
        # Run YOLOv8 inference
        results = self.model(image, conf=self.confidence_threshold, verbose=False)
        
        # Calculate inference time
        end_time = cv2.getTickCount()
        inference_time = (end_time - start_time) / cv2.getTickFrequency()
        self.total_inference_time += inference_time
        self.detection_count += 1
        
        detections = []
        
        for result in results:
            boxes = result.boxes
            
            if boxes is None:
                continue
                
            for i, box in enumerate(boxes):
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = box.conf[0].cpu().numpy()
                class_id = int(box.cls[0].cpu().numpy())
                
                # Filter for vehicle classes only
                if class_id in self.vehicle_classes:
                    detection = {
                        'bbox': (int(x1), int(y1), int(x2), int(y2)),
                        'confidence': float(confidence),
                        'class_id': class_id,
                        'class_name': self.vehicle_classes[class_id],
                        'center': (int((x1 + x2) / 2), int((y1 + y2) / 2)),
                        'area': int((x2 - x1) * (y2 - y1))
                    }
                    
                    # Add cropped image if requested
                    if return_crops:
                        crop = image[int(y1):int(y2), int(x1):int(x2)]
                        detection['crop'] = crop
                    
                    detections.append(detection)
        
        return detections
    
    def filter_detections_by_size(self, detections: List[Dict], 
                                 min_area: int = 1000,
                                 max_area: int = 50000) -> List[Dict]:
        """
        Filter detections by bounding box area
        
        Args:
            detections: List of vehicle detections
            min_area: Minimum bounding box area in pixels
            max_area: Maximum bounding box area in pixels
            
        Returns:
            Filtered list of detections
        """
        filtered = []
        for detection in detections:
            area = detection['area']
            if min_area <= area <= max_area:
                filtered.append(detection)
        return filtered
    
    def filter_detections_by_position(self, detections: List[Dict],
                                    image_height: int,
                                    road_region_ratio: float = 0.6) -> List[Dict]:
        """
        Filter detections to focus on road region (bottom part of image)
        
        Args:
            detections: List of vehicle detections
            image_height: Height of the input image
            road_region_ratio: Ratio of image height to consider as road region
            
        Returns:
            Filtered list of detections
        """
        road_threshold = int(image_height * (1 - road_region_ratio))
        
        filtered = []
        for detection in detections:
            bbox = detection['bbox']
            center_y = detection['center'][1]
            
            # Keep detections in the lower part of the image (road region)
            if center_y > road_threshold:
                filtered.append(detection)
        
        return filtered
    
    def non_max_suppression(self, detections: List[Dict], 
                           iou_threshold: float = 0.5) -> List[Dict]:
        """
        Apply Non-Maximum Suppression to remove overlapping detections
        
        Args:
            detections: List of vehicle detections
            iou_threshold: IoU threshold for suppression
            
        Returns:
            Filtered list of detections after NMS
        """
        if len(detections) == 0:
            return []
        
        # Extract bounding boxes and scores
        boxes = []
        scores = []
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            boxes.append([x1, y1, x2, y2])
            scores.append(detection['confidence'])
        
        boxes = np.array(boxes, dtype=np.float32)
        scores = np.array(scores, dtype=np.float32)
        
        # Apply OpenCV NMS
        indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), 
                                  self.confidence_threshold, iou_threshold)
        
        if len(indices) == 0:
            return []
        
        # Return filtered detections
        filtered_detections = []
        for i in indices.flatten():
            filtered_detections.append(detections[i])
        
        return filtered_detections
    
    def detect_and_filter(self, image: np.ndarray) -> List[Dict]:
        """
        Complete detection pipeline with filtering
        
        Args:
            image: Input image
            
        Returns:
            Filtered list of vehicle detections
        """
        # Detect vehicles
        detections = self.detect_vehicles(image)
        
        # Apply filters
        detections = self.filter_detections_by_size(detections)
        detections = self.filter_detections_by_position(detections, image.shape[0])
        detections = self.non_max_suppression(detections)
        
        return detections
    
    def visualize_detections(self, image: np.ndarray, 
                           detections: List[Dict],
                           show_confidence: bool = True,
                           show_class: bool = True) -> np.ndarray:
        """
        Visualize vehicle detections on the image
        
        Args:
            image: Input image
            detections: List of vehicle detections
            show_confidence: Whether to show confidence scores
            show_class: Whether to show class names
            
        Returns:
            Image with detection visualizations
        """
        vis_image = image.copy()
        
        # Color map for different vehicle types
        color_map = {
            'car': (0, 255, 0),      # Green
            'motorcycle': (255, 0, 0), # Blue  
            'bus': (0, 255, 255),    # Yellow
            'truck': (255, 0, 255)   # Magenta
        }
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class_name']
            
            # Get color for this vehicle type
            color = color_map.get(class_name, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
            
            # Draw center point
            center = detection['center']
            cv2.circle(vis_image, center, 4, color, -1)
            
            # Prepare label text
            label_parts = []
            if show_class:
                label_parts.append(class_name)
            if show_confidence:
                label_parts.append(f"{confidence:.2f}")
            
            if label_parts:
                label = " ".join(label_parts)
                
                # Get text size for background rectangle
                (text_width, text_height), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                
                # Draw background rectangle
                cv2.rectangle(vis_image, 
                            (x1, y1 - text_height - 10),
                            (x1 + text_width + 10, y1),
                            color, -1)
                
                # Draw text
                cv2.putText(vis_image, label,
                          (x1 + 5, y1 - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        return vis_image
    
    def get_detection_stats(self) -> Dict:
        """
        Get detection performance statistics
        
        Returns:
            Dictionary with performance metrics
        """
        avg_inference_time = (self.total_inference_time / self.detection_count 
                             if self.detection_count > 0 else 0)
        
        return {
            'total_detections': self.detection_count,
            'total_inference_time': self.total_inference_time,
            'average_inference_time': avg_inference_time,
            'fps': 1.0 / avg_inference_time if avg_inference_time > 0 else 0,
            'device': self.device,
            'confidence_threshold': self.confidence_threshold
        }
    
    def reset_stats(self):
        """Reset detection statistics"""
        self.detection_count = 0
        self.total_inference_time = 0.0
    
    def update_confidence_threshold(self, new_threshold: float):
        """Update confidence threshold for detections"""
        self.confidence_threshold = max(0.1, min(0.9, new_threshold))
        print(f"Updated confidence threshold to: {self.confidence_threshold}")
    
    def get_vehicle_bottom_points(self, detections: List[Dict]) -> List[Tuple[float, float]]:
        """
        Get bottom-center points of detected vehicles for tracking
        
        Args:
            detections: List of vehicle detections
            
        Returns:
            List of bottom-center points (x, y)
        """
        bottom_points = []
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            bottom_center_x = (x1 + x2) / 2
            bottom_center_y = y2  # Bottom of bounding box
            bottom_points.append((bottom_center_x, bottom_center_y))
        
        return bottom_points