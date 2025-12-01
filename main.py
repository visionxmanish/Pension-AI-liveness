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

    # Save temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        # 1. Validate Quality
        is_valid, msg = validate_image_quality(tmp_path)
        if not is_valid:
            raise HTTPException(status_code=400, detail=f"Image quality check failed: {msg}")

        # 2. Generate Embedding
        embedding = get_face_embedding(tmp_path)
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
        # Should we generate a meet link? Probably yes, to handle "I can't log in" cases.
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

    # Save temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

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
