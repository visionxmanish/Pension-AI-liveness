from pydantic import BaseModel
from typing import List, Optional

class UserBase(BaseModel):
    pension_id: str
    name: str

class UserCreate(UserBase):
    pass

class UserResponse(UserBase):
    id: int
    is_active: bool

    class Config:
        orm_mode = True

class VerificationResponse(BaseModel):
    status: str
    message: str
    meet_link: Optional[str] = None
