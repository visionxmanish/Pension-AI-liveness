from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session
from database import engine, Base, get_db
import models
import schemas
import shutil
import os
import tempfile
from services.face_service import get_face_embedding, validate_image_quality, verify_face, check_liveness
from services.utils import generate_meet_link
import json

from PIL import Image, ExifTags
import pillow_heif
pillow_heif.register_heif_opener()

def process_uploaded_image(upload_file: UploadFile) -> str:
    """ Processes upload, standardizes format/orientation, compresses large files, returns temp path """
    upload_file.file.seek(0)
    img = Image.open(upload_file.file)
    
    try:
        # Handle EXIF orientation (crucial for mobile photos)
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        exif = img._getexif()
        if exif is not None:
            exif_orientation = exif.get(orientation, 1)
            if exif_orientation == 3:
                img = img.rotate(180, expand=True)
            elif exif_orientation == 6:
                img = img.rotate(270, expand=True)
            elif exif_orientation == 8:
                img = img.rotate(90, expand=True)
    except Exception:
        pass

    # Convert alpha channels/HEIC to standard RGB
    if img.mode != 'RGB':
        img = img.convert('RGB')
        
    # Resize securely to prevent OOM errors on massive phone camera pics
    img.thumbnail((2000, 2000), Image.Resampling.LANCZOS)
    
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    img.save(tmp.name, format="JPEG", quality=90)
    tmp.close()
    return tmp.name

# Create tables
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Pension Face Recognition API")

@app.post("/register", response_model=schemas.UserResponse)
async def register_user(
    pension_id: str = Form(...),
    name: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    # Check if user already exists
    db_user = db.query(models.User).filter(models.User.pension_id == pension_id).first()
    if db_user:
        raise HTTPException(status_code=400, detail="User with this Pension ID already registered")

    try:
        tmp_path = process_uploaded_image(file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image format or file corrupted: {str(e)}")

    try:
        # 1. Validate Quality
        is_valid, msg = validate_image_quality(tmp_path)
        if not is_valid:
            raise HTTPException(status_code=400, detail=f"Image quality check failed: {msg}")

        # 2. Generate Embedding
        try:
            embedding = get_face_embedding(tmp_path, require_single=True)
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=str(ve))
            
        if embedding is None:
            raise HTTPException(status_code=400, detail="No face detected in the image")

        # 3. Save to DB
        new_user = models.User(pension_id=pension_id, name=name)
        db.add(new_user)
        db.commit()
        db.refresh(new_user)

        new_face = models.FaceEncoding(user_id=new_user.id, embedding=embedding)
        db.add(new_face)
        db.commit()

        return new_user

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

@app.post("/verify", response_model=schemas.VerificationResponse)
async def verify_user(
    pension_id: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    # Fetch user
    user = db.query(models.User).filter(models.User.pension_id == pension_id).first()
    if not user:
        # If user not found, we can't verify. 
        return schemas.VerificationResponse(
            status="failure",
            message="User not found",
            meet_link=generate_meet_link()
        )

    # Fetch stored embedding
    # Assuming one face per user for now
    if not user.face_encodings:
         return schemas.VerificationResponse(
            status="failure",
            message="No registered face data found for this user",
            meet_link=generate_meet_link()
        )
    
    stored_embedding = user.face_encodings[0].embedding

    try:
        tmp_path = process_uploaded_image(file)
    except Exception as e:
        return schemas.VerificationResponse(
            status="failure",
            message=f"Invalid image format or file corrupted: {str(e)}",
            meet_link=generate_meet_link()
        )

    try:
        # 1. Validate Quality (Anti-glare, Blur)
        is_valid, msg = validate_image_quality(tmp_path)
        if not is_valid:
             return schemas.VerificationResponse(
                status="failure",
                message=f"Image quality check failed: {msg}",
                meet_link=generate_meet_link()
            )

        # 2. Liveness Check (Anti-Spoofing)
        is_real, liveness_msg = check_liveness(tmp_path)
        if not is_real:
             return schemas.VerificationResponse(
                status="failure",
                message=liveness_msg,
                meet_link=generate_meet_link()
            )

        # 3. Verify Face
        is_match, match_msg = verify_face(tmp_path, stored_embedding)
        
        if is_match:
            return schemas.VerificationResponse(
                status="success",
                message="Verification successful"
            )
        else:
            return schemas.VerificationResponse(
                status="failure",
                message=f"Verification failed: {match_msg}",
                meet_link=generate_meet_link()
            )

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

from services.liveness_service import verify_video_liveness

@app.post("/verify_liveness_video", response_model=schemas.VerificationResponse)
async def verify_liveness_video(
    pension_id: str = Form(...),
    verification_type: str = Form(...), # e.g., "smile", "move_head_left", "move_head_right"
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):

    # if file is not video, return failure
    if not file.filename.endswith(".mp4"):
        return schemas.VerificationResponse(
            status="failure",
            message="File is not a video",
            meet_link=generate_meet_link()
        )

    # # save to temp file 
    # with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
    #     shutil.copyfileobj(file.file, tmp)
    #     tmp_path = tmp.name
    
    # Fetch user
    user = db.query(models.User).filter(models.User.pension_id == pension_id).first()
    if not user:
        return schemas.VerificationResponse(
            status="failure",
            message="User not found",
            meet_link=generate_meet_link()
        )

    # Fetch stored embedding
    if not user.face_encodings:
         return schemas.VerificationResponse(
            status="failure",
            message="No registered face data found for this user",
            meet_link=generate_meet_link()
        )
    
    stored_embedding = user.face_encodings[0].embedding

    # Save temp video file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_video:
        shutil.copyfileobj(file.file, tmp_video)
        tmp_video_path = tmp_video.name

    best_frame_path = None
    try:
        # Map verification_type to required_actions
        # Supported types: smile, move_head_left, move_head_right
        actions_map = {
            "smile": "smile",
            "move_head_left": "turn_left",
            "move_head_right": "turn_right"
        }
        
        # Allow comma-separated values
        requested_types = [t.strip() for t in verification_type.split(",")]
        required_actions = []
        
        for rt in requested_types:
            if rt in actions_map:
                required_actions.append(actions_map[rt])
            else:
                # Fallback or ignore? Let's just pass it if it matches internal names
                if rt in ["turn_left", "turn_right", "smile"]:
                     required_actions.append(rt)
        
        if not required_actions:
             # Default if invalid type provided
             required_actions = ["smile"] 

        # 1. Verify Liveness (Video Actions)
        is_live, liveness_msg, best_frame_path = verify_video_liveness(
            tmp_video_path, 
            required_actions=required_actions
        )
        
        if not is_live:
             return schemas.VerificationResponse(
                status="failure",
                message=liveness_msg,
                meet_link=generate_meet_link()
            )

        # 2. Verify Face (using the best frame from video)
        if best_frame_path:
            is_match, match_msg = verify_face(best_frame_path, stored_embedding)
            
            if is_match:
                return schemas.VerificationResponse(
                    status="success",
                    message="Verification successful"
                )
            else:
                return schemas.VerificationResponse(
                    status="failure",
                    message=f"Verification failed: {match_msg}",
                    meet_link=generate_meet_link()
                )
        else:
             return schemas.VerificationResponse(
                status="failure",
                message="Could not extract a valid frame for face verification",
                meet_link=generate_meet_link()
            )

    except Exception as e:
        return schemas.VerificationResponse(
            status="failure",
            message=f"System error: {str(e)}",
            meet_link=generate_meet_link()
        )

    finally:
        if os.path.exists(tmp_video_path):
            os.remove(tmp_video_path)
        if best_frame_path and os.path.exists(best_frame_path):
            os.remove(best_frame_path)
