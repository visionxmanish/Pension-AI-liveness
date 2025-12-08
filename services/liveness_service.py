import cv2
import numpy as np
import math
import os

class LivenessDetector:
    def __init__(self):
        self.facemark = cv2.face.createFacemarkLBF()
        model_path = "lbfmodel.yaml"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} not found. Please download it.")
        self.facemark.loadModel(model_path)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")

    def get_landmarks(self, image):
        """
        Processes the image and returns face landmarks (68 points).
        Returns None if no face or landmarks found.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) > 0:
            success, landmarks = self.facemark.fit(image, faces)
            if success:
                return landmarks[0][0] # Return first face's landmarks
        return None

    def calculate_pose(self, landmarks, image_shape):
        """
        Calculates head pose (pitch, yaw, roll) from face landmarks.
        Returns (pitch, yaw, roll) in degrees.
        """
        img_h, img_w, img_c = image_shape
        
        # 2D image points. If you change the image, you need to change vector
        image_points = np.array([
            landmarks[30],     # Nose tip
            landmarks[8],      # Chin
            landmarks[36],     # Left eye left corner
            landmarks[45],     # Right eye right corner
            landmarks[48],     # Left Mouth corner
            landmarks[54]      # Right mouth corner
        ], dtype="double")

        # 3D model points.
        model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corner
            (-150.0, -150.0, -125.0),    # Left Mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ])

        # Camera internals
        focal_length = img_w
        center = (img_w/2, img_h/2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]], dtype = "double"
        )

        dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
        
        (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

        # Calculate Euler angles
        rmat, jac = cv2.Rodrigues(rotation_vector)
        # Get angles
        # Returns: (eulerAngles, mtxR, mtxQ, Qx, Qy, Qz)
        ret = cv2.RQDecomp3x3(rmat)
        angles = ret[0]

        x = angles[0]
        y = angles[1]
        z = angles[2]
        
        # Adjust angles to be intuitive
        # Pitch: Up/Down (x)
        # Yaw: Left/Right (y)
        # Roll: Tilt (z)

        return x, y, z

    def detect_expression(self, landmarks):
        """
        Detects expressions like smile.
        Returns a dictionary of detected expressions.
        """
        # Smile detection: Ratio of mouth width to face width or inter-ocular distance
        # 68 landmarks:
        # Mouth: 48-67
        # Left corner: 48, Right corner: 54
        # Left eye: 36-41, Right eye: 42-47
        # Outer corners: 36, 45
        
        left_mouth = landmarks[48]
        right_mouth = landmarks[54]
        
        left_eye = landmarks[36]
        right_eye = landmarks[45]
        
        mouth_width = math.sqrt((left_mouth[0] - right_mouth[0])**2 + (left_mouth[1] - right_mouth[1])**2)
        eye_dist = math.sqrt((left_eye[0] - right_eye[0])**2 + (left_eye[1] - right_eye[1])**2)
        
        smile_ratio = mouth_width / eye_dist
        
        # Threshold for smile (needs tuning)
        is_smiling = smile_ratio > 0.55 
        
        return {"smile": is_smiling, "ratio": smile_ratio}

def verify_video_liveness(video_path, required_actions=["turn_left", "turn_right", "smile"]):
    """
    Processes a video to verify if the user performs the required actions.
    Returns (success, message, best_frame_path)
    """
    cap = cv2.VideoCapture(video_path)
    try:
        detector = LivenessDetector()
    except FileNotFoundError as e:
        return False, str(e), None
    
    action_status = {action: False for action in required_actions}
    best_frame = None
    max_face_quality = -float('inf')
    
    # Thresholds
    YAW_THRESHOLD = 15 # Degrees
    
    frame_count = 0
    
    import tempfile
    import os

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            
            frame_count += 1
            if frame_count % 3 != 0:
                continue

            landmarks = detector.get_landmarks(frame)
            
            if landmarks is not None:
                pitch, yaw, roll = detector.calculate_pose(landmarks, frame.shape)
                expressions = detector.detect_expression(landmarks)
                
                # Check actions
                if "turn_left" in required_actions and not action_status["turn_left"]:
                    if yaw < -YAW_THRESHOLD: 
                        action_status["turn_left"] = True
                        print("Detected Turn Left")

                if "turn_right" in required_actions and not action_status["turn_right"]:
                    if yaw > YAW_THRESHOLD:
                        action_status["turn_right"] = True
                        print("Detected Turn Right")
                        
                if "smile" in required_actions and not action_status["smile"]:
                    if expressions["smile"]:
                        action_status["smile"] = True
                        print("Detected Smile")
                
                # Capture best frontal frame
                score = 100 - (abs(yaw) + abs(pitch))
                if score > max_face_quality:
                    max_face_quality = score
                    best_frame = frame.copy()

        cap.release()
        
        missing_actions = [action for action, done in action_status.items() if not done]
        
        if not missing_actions:
            if best_frame is not None and best_frame.size > 0:
                print(f"Saving best frame. Shape: {best_frame.shape}")
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                    cv2.imwrite(tmp.name, best_frame)
                    return True, "Liveness verified", tmp.name
            else:
                print(f"Best frame is invalid. None: {best_frame is None}, Size: {best_frame.size if best_frame is not None else 'N/A'}")
                return False, "Liveness verified but could not extract a valid face frame.", None
        else:
            return False, f"Liveness check failed. Missing actions: {', '.join(missing_actions)}", None

    except Exception as e:
        print(f"Error in video processing: {e}")
        return False, f"Error processing video: {str(e)}", None
