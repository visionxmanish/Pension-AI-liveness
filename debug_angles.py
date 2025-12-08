import cv2
import numpy as np
import math

# Create a rotation matrix for 90 degrees around X axis
theta = np.radians(90)
c, s = np.cos(theta), np.sin(theta)
R_x = np.array([
    [1, 0, 0],
    [0, c, -s],
    [0, s, c]
])

# Decompose
ret = cv2.RQDecomp3x3(R_x)
angles = ret[0] # (pitch, yaw, roll) usually (x, y, z)

print(f"Input: 90 degrees X rotation")
print(f"Output angles: {angles}")
