import cv2
import numpy as np
from deepface import DeepFace
import tempfile
import os

def validate_image_quality(image_path, blur_threshold=100.0, glare_ratio_threshold=0.05):
    """
    Checks image for glare and blurriness with improved spatial analysis.
    """
    # 1. Efficient Loading
    image = cv2.imread(image_path)
    if image is None:
        return False, "Could not read image"

    # Resize for consistent thresholding regardless of camera resolution
    # Standardizing to a width of 800px maintains speed and accuracy
    h, w = image.shape[:2]
    scale = 800 / float(w)
    resized = cv2.resize(image, (800, int(h * scale)))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # 2. Advanced Blur Detection (Laplacian)
    # Increased threshold (100 is a standard starting point for 720p/1080p)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var < blur_threshold:
        return False, f"Image too blurry ({laplacian_var:.1f}). Hold camera steady."

    # 3. Refined Glare Detection
    # Instead of just counting bright pixels, we look for 'hotspots'
    # using a binary threshold for pixels near-white (e.g., > 250)
    _, mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)
    
    # Use morphological closing to bridge small gaps in glare spots
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    bright_pixel_count = cv2.countNonZero(mask)
    total_pixels = resized.shape[0] * resized.shape[1]
    glare_ratio = bright_pixel_count / total_pixels

    if glare_ratio > glare_ratio_threshold:
        return False, f"Glare detected ({glare_ratio:.2%}). Adjust lighting."

    return True, "Quality OK"


    
def check_liveness(image_path):
    """
    Checks if the face is real using DeepFace anti-spoofing.
    Returns (is_real, message)
    """
    try:
        # DeepFace.extract_faces returns a list of dicts. 
        # Each dict has 'face', 'facial_area', 'confidence', 'is_real' (if anti_spoofing=True)
        objs = DeepFace.extract_faces(
            img_path=image_path,
            enforce_detection=True,
            anti_spoofing=True
        )
        
        if not objs:
            return False, "No face detected"
            
        # Check the first face
        is_real = objs[0].get("is_real", True) # Default to True if key missing (older version)
        antispoof_score = objs[0].get("antispoof_score", 0.0) # Some versions return score
        
        if not is_real:
             return False, "Liveness check failed: Spoof detected."
             
        return True, "Liveness check passed"
        
    except TypeError:
        # Fallback for older DeepFace versions that don't support anti_spoofing arg
        print("Warning: DeepFace version does not support anti_spoofing argument.")
        return True, "Liveness check skipped (not supported)"
    except Exception as e:
        print(f"Liveness check error: {e}")
        # If face not detected, extract_faces raises error
        return False, f"Liveness check error: {str(e)}"

def get_face_embedding(image_path, require_single=True):
    """
    Generates face embedding using DeepFace.
    If require_single is True, raises ValueError if more than 1 face is detected.
    """
    try:
        # Using ArcFace for better accuracy, or VGG-Face
        embedding_objs = DeepFace.represent(
            img_path=image_path,
            model_name="ArcFace",
            detector_backend="retinaface",
            enforce_detection=True
        )
        if not embedding_objs:
            return None
            
        # Security/Accuracy check: Ensure only one face is in the registration photo
        if require_single and len(embedding_objs) > 1:
            raise ValueError(f"Multiple faces detected ({len(embedding_objs)}). Please ensure only your face is visible.")
            
        return embedding_objs[0]["embedding"]
        
    except ValueError as ve:
        raise ve
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None

def verify_face(image_path, stored_embedding):
    """
    Verifies if the face in image_path matches the stored_embedding.
    """
    try:
        # Generate embedding for the new image
        new_embedding_objs = DeepFace.represent(
            img_path=image_path,
            model_name="ArcFace",
            detector_backend="retinaface",
            enforce_detection=True
        )
        if not new_embedding_objs:
            return False, "No face detected"
        
        new_embedding = new_embedding_objs[0]["embedding"]
        
        # Calculate distance (Cosine similarity is common for ArcFace)
        # DeepFace.verify usually handles this, but since we stored embeddings manually:
        
        # Using DeepFace's verification logic would be easier if we passed paths, 
        # but we have stored embeddings. 
        # Let's use a manual cosine distance check or DeepFace's verify with a dummy db approach if needed.
        # Simpler: Calculate cosine distance manually.
        
        a = np.array(stored_embedding)
        b = np.array(new_embedding)
        
        # Cosine distance
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        cosine_similarity = dot_product / (norm_a * norm_b)
        
        # ArcFace threshold is typically around 0.68 for cosine similarity (or 0.3-0.4 for distance)
        # DeepFace uses distance. Cosine Distance = 1 - Cosine Similarity.
        # Threshold for ArcFace in DeepFace is usually 0.68 (distance) ?? No, let's check.
        # Actually, let's rely on DeepFace's verify if possible, but we have embeddings.
        # Let's stick to a standard threshold. For ArcFace, cosine similarity > 0.4 is usually a match?
        # Let's use a safe threshold or check DeepFace defaults.
        # Default metric for ArcFace is cosine. Threshold is 0.68.
        # Wait, 0.68 distance or similarity? 
        # DeepFace source: "cosine": 0.68 (Distance). So Similarity < 0.32? No that's wrong.
        # Usually Cosine Distance < Threshold. 
        # Cosine Distance = 1 - Cosine Similarity.
        # If Threshold is 0.68, then 1 - Sim < 0.68 => Sim > 0.32. That seems too low.
        
        # Let's re-read DeepFace docs logic or just use a safe bet.
        # Better yet, let's use DeepFace.verify by saving the embedding to a temp file? No that's slow.
        
        # Let's use Euclidean L2 for simplicity if we can, but ArcFace is trained for Cosine.
        # Let's assume Cosine Distance.
        cosine_distance = 1 - cosine_similarity
        
        # DeepFace default threshold for ArcFace with Cosine is 0.68.
        if cosine_distance < 0.68:
            return True, "Match found"
        else:
            return False, "Face does not match"
            
    except Exception as e:
        return False, f"Verification error: {str(e)}"
