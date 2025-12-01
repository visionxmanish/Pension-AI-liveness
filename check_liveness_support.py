from deepface import DeepFace
import cv2
import numpy as np
import os

# Create a dummy image
img = np.zeros((500, 500, 3), dtype=np.uint8)
cv2.imwrite("dummy_check.jpg", img)

try:
    # Try to call extract_faces with anti_spoofing
    objs = DeepFace.extract_faces(img_path="dummy_check.jpg", anti_spoofing=True)
    print("Anti-spoofing supported!")
except TypeError:
    print("Anti-spoofing NOT supported in this version.")
except Exception as e:
    print(f"Other error: {e}")
finally:
    if os.path.exists("dummy_check.jpg"):
        os.remove("dummy_check.jpg")
