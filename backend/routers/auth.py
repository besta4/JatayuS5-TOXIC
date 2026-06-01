"""
routers/auth.py — Authentication endpoints.

Handles user registration, login, logout, and token refresh.
"""

from fastapi import APIRouter, HTTPException, status, Request, Response, Depends
from datetime import datetime, timezone

import database as db
from auth.models import (
    RegisterRequest, LoginRequest, TokenResponse, UserResponse,
    RefreshTokenRequest, ChangePasswordRequest
)
from auth.password import hash_password, verify_password
from auth.jwt_handler import (
    create_access_token, create_refresh_token, verify_refresh_token, get_token_hash
)
from auth.dependencies import get_current_user, get_current_user_including_restricted, get_client_info
from auth.models import TokenData
from database import UserType, AccountType

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def register(request: Request, body: RegisterRequest):
    """
    Register a new user account.

    - Creates user with specified type (CUSTOMER, MERCHANT)
    - Creates primary account with initial balance
    - Returns JWT tokens for immediate login
    """
    # Check if email already exists
    existing = db.get_user_by_email(body.email)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email already registered"
        )

    # Validate merchant has business name
    if body.user_type == "MERCHANT" and not body.business_name:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Business name required for merchant accounts"
        )

    # Admin registration not allowed via API
    if body.user_type == "ADMIN":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin accounts cannot be created via registration"
        )

    # Hash password and create user
    password_hashed = hash_password(body.password)

    try:
        user_id = db.create_user(
            email=body.email,
            password_hash=password_hashed,
            user_type=UserType(body.user_type),
            display_name=body.display_name,
            phone=body.phone,
            business_name=body.business_name
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create user: {str(e)}"
        )

    # Create primary account
    account_type = AccountType.MERCHANT if body.user_type == "MERCHANT" else AccountType.SAVINGS
    initial_balance = 10000.0  # Demo: start with ₹10,000
    db.create_account(user_id, account_type, initial_balance, is_primary=True)

    # Get client info for session
    client_info = get_client_info(request)

    # Create session and tokens
    session_id = db.create_session(
        user_id=user_id,
        token_hash="pending",
        device_id=client_info["device_id"],
        ip_address=client_info["ip_address"],
        user_agent=client_info["user_agent"]
    )

    access_token = create_access_token(
        user_id=user_id,
        user_type=body.user_type,
        session_id=session_id,
        device_id=client_info["device_id"]
    )

    refresh_token = create_refresh_token(user_id, session_id)

    # Register device
    db.register_device(
        user_id=user_id,
        device_id=client_info["device_id"],
        device_type="WEB"
    )

    # Update login timestamp
    db.update_user_login(user_id)

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        user=UserResponse(
            user_id=user_id,
            email=body.email,
            display_name=body.display_name,
            user_type=body.user_type,
            account_status="ACTIVE",
            business_name=body.business_name,
            phone=body.phone,
            created_at=datetime.now(timezone.utc).isoformat()
        )
    )


@router.post("/login", response_model=TokenResponse)
async def login(request: Request, body: LoginRequest, response: Response):
    """
    Login with email and password.

    Returns JWT access and refresh tokens.
    """
    # Find user
    user = db.get_user_by_email(body.email)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )

    # Check if account is locked
    if user.get("locked_until"):
        locked_until = datetime.fromisoformat(user["locked_until"])
        if datetime.now(timezone.utc) < locked_until:
            raise HTTPException(
                status_code=status.HTTP_423_LOCKED,
                detail=f"Account locked until {locked_until.isoformat()}"
            )

    # Check account status — BLOCKED/SUSPENDED users can still log in
    # but they will be redirected to the /suspended page where they can
    # ONLY contact support. Blocking at the login layer prevents them from
    # even reaching the support UI, which defeats the purpose.
    restricted = user["account_status"] in ("BLOCKED", "SUSPENDED")
    # Verify password
    if not verify_password(body.password, user["password_hash"]):
        # Increment failed attempts
        failed = db.increment_failed_login(user["user_id"])
        if failed >= 5:
            db.lock_user(user["user_id"], minutes=30)
            raise HTTPException(
                status_code=status.HTTP_423_LOCKED,
                detail="Account locked due to too many failed attempts"
            )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )

    # Get client info
    client_info = get_client_info(request)
    device_id = body.device_id or client_info["device_id"]

    # Create session
    session_id = db.create_session(
        user_id=user["user_id"],
        token_hash="pending",
        device_id=device_id,
        ip_address=client_info["ip_address"],
        user_agent=client_info["user_agent"]
    )

    # Create tokens
    access_token = create_access_token(
        user_id=user["user_id"],
        user_type=user["user_type"],
        session_id=session_id,
        device_id=device_id
    )

    refresh_token = create_refresh_token(user["user_id"], session_id)

    # Register device
    db.register_device(
        user_id=user["user_id"],
        device_id=device_id,
        device_type="WEB"
    )

    # Update login timestamp
    db.update_user_login(user["user_id"])

    # Get profile
    profile = db.get_user_with_profile(user["user_id"])

    # Set device cookie
    response.set_cookie(
        key="device_id",
        value=device_id,
        max_age=60 * 60 * 24 * 365,  # 1 year
        httponly=True,
        samesite="lax"
    )

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        user=UserResponse(
            user_id=user["user_id"],
            email=user["email"],
            display_name=profile.get("display_name", "User") if profile else "User",
            user_type=user["user_type"],
            account_status=user["account_status"],
            business_name=profile.get("business_name") if profile else None,
            phone=user.get("phone"),
            created_at=user["created_at"],
            last_login=user.get("last_login")
        )
    )


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(body: RefreshTokenRequest):
    """
    Refresh access token using refresh token.
    """
    # Verify refresh token
    data = verify_refresh_token(body.refresh_token)
    if not data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token"
        )

    # Get user
    user = db.get_user_by_id(data["user_id"])
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )

    # Check session is still valid
    session = db.get_session(data["session_id"])
    if not session or not session.get("is_active"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session has been invalidated"
        )

    # Create new tokens
    access_token = create_access_token(
        user_id=user["user_id"],
        user_type=user["user_type"],
        session_id=data["session_id"],
        device_id=session.get("device_id")
    )

    new_refresh_token = create_refresh_token(user["user_id"], data["session_id"])

    profile = db.get_user_with_profile(user["user_id"])

    return TokenResponse(
        access_token=access_token,
        refresh_token=new_refresh_token,
        user=UserResponse(
            user_id=user["user_id"],
            email=user["email"],
            display_name=profile.get("display_name", "User") if profile else "User",
            user_type=user["user_type"],
            account_status=user["account_status"],
            business_name=profile.get("business_name") if profile else None,
            phone=user.get("phone"),
            created_at=user["created_at"],
            last_login=user.get("last_login")
        )
    )


@router.post("/logout")
async def logout(user: TokenData = Depends(get_current_user_including_restricted)):
    """
    Logout and invalidate current session.
    """
    if user.session_id:
        db.invalidate_session(user.session_id)

    return {"message": "Logged out successfully"}


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(user: TokenData = Depends(get_current_user)):
    """
    Get current user's profile information.
    """
    profile = db.get_user_with_profile(user.user_id)
    if not profile:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    return UserResponse(
        user_id=profile["user_id"],
        email=profile["email"],
        display_name=profile.get("display_name", "User"),
        user_type=profile["user_type"],
        account_status=profile["account_status"],
        business_name=profile.get("business_name"),
        phone=profile.get("phone"),
        created_at=profile["created_at"],
        last_login=profile.get("last_login")
    )


@router.put("/password")
async def change_password(
    body: ChangePasswordRequest,
    user: TokenData = Depends(get_current_user)
):
    """
    Change current user's password.
    """
    # Get user
    db_user = db.get_user_by_id(user.user_id)
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")

    # Verify current password
    if not verify_password(body.current_password, db_user["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Current password is incorrect"
        )

    # Update password
    new_hash = hash_password(body.new_password)
    db.update_user(user.user_id, password_hash=new_hash)

    # Invalidate all other sessions
    db.invalidate_all_sessions(user.user_id)

    return {"message": "Password changed successfully. Please login again."}
