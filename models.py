from sqlalchemy import Column, Integer, String, Boolean, ForeignKey, JSON
from sqlalchemy.orm import relationship
from database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    pension_id = Column(String, unique=True, index=True)
    name = Column(String)
    is_active = Column(Boolean, default=True)

    face_encodings = relationship("FaceEncoding", back_populates="user")

class FaceEncoding(Base):
    __tablename__ = "face_encodings"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    embedding = Column(JSON)  # Storing embedding as a JSON list

    user = relationship("User", back_populates="face_encodings")
