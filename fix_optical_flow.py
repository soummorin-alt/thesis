#!/usr/bin/env python3
"""
Fix script for optical flow function calls
"""

import os

# Read the file
with open('src/flow/optical_flow.py', 'r') as f:
    content = f.read()

# Fix the function calls
content = content.replace(
    'cv2.calcOpticalFlowPyrLK(gray1, gray2, **self.flow_params)',
    'cv2.calcOpticalFlowPyrLK(gray1, gray2, **self.flow_params)'
)

# Write back
with open('src/flow/optical_flow.py', 'w') as f:
    f.write(content)

print("Fixed optical flow function calls")