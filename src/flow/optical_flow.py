"""
Optical Flow Computation - Fixed Version
"""
import cv2
import numpy as np
import torch
from typing import Dict, List, Tuple

class OpticalFlowEstimator:
    def __init__(self, method="farneback", use_gpu=True):
        self.method = method
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        self.raft_model = None
        print(f"Optical flow: {method} on {self.device}")
    
    def compute_dense_flow(self, frame1, frame2):
        try:
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowPyrLK(gray1, gray2, 0.5, 3, 15, 3, 5, 1.2, 0)
            return flow
        except Exception as e:
            print(f"Flow error: {e}")
            h, w = frame1.shape[:2]
            return np.zeros((h, w, 2), dtype=np.float32)
    
    def extract_vehicle_flow(self, flow, bbox):
        try:
            x1, y1, x2, y2 = bbox
            h, w = flow.shape[:2]
            x1, y1, x2, y2 = max(0,x1), max(0,y1), min(w,x2), min(h,y2)
            if x2 <= x1 or y2 <= y1:
                return {"mean_flow": (0.0, 0.0), "magnitude": 0.0, "angle": 0.0, "flow_vectors": 0}
            roi = flow[y1:y2, x1:x2]
            if roi.size == 0:
                return {"mean_flow": (0.0, 0.0), "magnitude": 0.0, "angle": 0.0, "flow_vectors": 0}
            fx, fy = np.mean(roi[:,:,0]), np.mean(roi[:,:,1])
            mag = np.sqrt(fx*fx + fy*fy)
            return {"mean_flow": (float(fx), float(fy)), "magnitude": float(mag), "angle": float(np.arctan2(fy,fx)), "flow_vectors": roi.size//2}
        except:
            return {"mean_flow": (0.0, 0.0), "magnitude": 0.0, "angle": 0.0, "flow_vectors": 0}
    
    def visualize_vehicle_flow(self, image, vehicles, flow):
        vis = image.copy()
        for v in vehicles:
            x1,y1,x2,y2 = v["bbox"]
            stats = self.extract_vehicle_flow(flow, (x1,y1,x2,y2))
            cv2.rectangle(vis, (x1,y1), (x2,y2), (0,255,0), 2)
            cx, cy = (x1+x2)//2, (y1+y2)//2
            fx, fy = stats["mean_flow"]
            ex, ey = int(cx + fx*10), int(cy + fy*10)
            cv2.arrowedLine(vis, (cx,cy), (ex,ey), (0,0,255), 2)
            cv2.putText(vis, f"F:{stats['magnitude']:.1f}", (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        return vis
    
    def get_flow_statistics(self):
        return {"method": self.method, "device": self.device, "gpu_available": torch.cuda.is_available(), "raft_loaded": self.raft_model is not None}
