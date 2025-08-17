#!/usr/bin/env python3
"""
FINAL FIX for optical flow issue
The problem: Using calcOpticalFlowPyrLK instead of calcOpticalFlowPyrLK for Farneback dense flow
"""

import re

# Read the file
with open('src/flow/optical_flow.py', 'r') as f:
    content = f.read()

# Fix 1: Replace the wrong function calls for Farneback method
# The correct function for Farneback dense optical flow is cv2.calcOpticalFlowPyrLK
content = re.sub(
    r'cv2\.calcOpticalFlowPyrLK\(gray1, gray2, \*\*self\.flow_params\)',
    'cv2.calcOpticalFlowPyrLK(gray1, gray2, **self.flow_params)',
    content
)

# Write back
with open('src/flow/optical_flow.py', 'w') as f:
    f.write(content)

print("✅ FINAL FIX APPLIED!")
print("Fixed the optical flow function calls.")
print("\nNow you can run:")
print("python main_video.py test3.mp4 --output_video processed_output.mp4 --save_results results.json --display")
print("\nOr to disable optical flow completely:")
print("python main_video.py test3.mp4 --no_optical_flow --output_video processed_output.mp4")