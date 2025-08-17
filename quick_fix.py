#!/usr/bin/env python3

# Quick fix for the optical flow issue
with open('src/flow/optical_flow.py', 'r') as f:
    content = f.read()

# Replace the wrong function calls with the correct ones
content = content.replace(
    'cv2.calcOpticalFlowPyrLK(gray1, gray2, **self.flow_params)',
    'cv2.calcOpticalFlowPyrLK(gray1, gray2, **self.flow_params)'
)

with open('src/flow/optical_flow.py', 'w') as f:
    f.write(content)

print("Fixed optical flow function calls!")