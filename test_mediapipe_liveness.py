from services.liveness_service import LivenessDetector
import cv2

def test_liveness_init():
    try:
        detector = LivenessDetector()
        print("LivenessDetector with MediaPipe initialized successfully!")
    except Exception as e:
        print(f"Failed to initialize: {e}")

if __name__ == "__main__":
    test_liveness_init()
