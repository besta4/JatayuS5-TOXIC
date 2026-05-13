"""
auth/models.py — Pydantic models for authentication requests and responses.
"""

from pydantic import BaseModel, EmailStr, Field
from typing import Optional
from enum import Enum



class UserTypeEnum(str, Enum):
    CUSTOMER = "CUSTOMER"
    MERCHANT = "MERCHANT"
    ADMIN = "ADMIN"




# ══════════════════════════════════════════════════════════════════════════════
# Request Models
# ══════════════════════════════════════════════════════════════════════════════

class RegisterRequest(BaseModel):
    """Registration request for new users."""
    email: EmailStr
    password: str = Field(..., min_length=6, max_length=128)
    display_name: str = Field(..., min_length=2, max_length=100)
    user_type: UserTypeEnum = UserTypeEnum.CUSTOMER
    phone: Optional[str] = Field(None, pattern=r"^\+?[0-9]{10,15}$")
    business_name: Optional[str] = None  # Required for MERCHANT

    class Config:
        json_schema_extra = {
            "example": {
                "email": "alice@example.com",
                "password": "securepass123",
                "display_name": "Alice Johnson",
                "user_type": "CUSTOMER",
                "phone": "+919876543210"
            }
        }


class LoginRequest(BaseModel):
    """Login request."""
    email: EmailStr
    password: str
    device_id: Optional[str] = None  # Browser fingerprint or generated ID

    class Config:
        json_schema_extra = {
            "example": {
                "email": "alice@demo.com",
                "password": "demo123"
            }
        }


class RefreshTokenRequest(BaseModel):
    """Refresh token request."""
    refresh_token: str


class ChangePasswordRequest(BaseModel):
    """Change password request."""
    current_password: str
    new_password: str = Field(..., min_length=6, max_length=128)


# ══════════════════════════════════════════════════════════════════════════════
# Response Models
# ══════════════════════════════════════════════════════════════════════════════

class TokenResponse(BaseModel):
    """JWT token response after login."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int = 86400  # 24 hours in seconds
    user: "UserResponse"


class UserResponse(BaseModel):
    """User information in responses."""
    user_id: str
    email: str
    display_name: str
    user_type: UserTypeEnum
    account_status: str
    business_name: Optional[str] = None
    phone: Optional[str] = None
    created_at: str
    last_login: Optional[str] = None


class AccountResponse(BaseModel):
    """Account information."""
    account_id: str
    account_type: str
    balance: float
    currency: str = "INR"
    is_primary: bool
    daily_limit: Optional[float] = None


class UserWithAccountsResponse(BaseModel):
    """User with their accounts."""
    user: UserResponse
    accounts: list[AccountResponse]


# ══════════════════════════════════════════════════════════════════════════════
# Internal Models
# ══════════════════════════════════════════════════════════════════════════════

class TokenData(BaseModel):
    """Data extracted from JWT token."""
    user_id: str
    user_type: str
    session_id: str
    device_id: Optional[str] = None


# Update forward references
TokenResponse.model_rebuild()
