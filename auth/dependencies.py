"""
auth/dependencies.py — FastAPI dependencies for authentication.

Provides dependency injection for protected routes.
"""

from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional, List
import uuid

from auth.jwt_handler import verify_token, get_token_hash
from auth.models import TokenData
from auth.rbac import can_access_route
import database as db

# HTTP Bearer token security scheme
security = HTTPBearer(auto_error=False)


async def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> TokenData:
    """
    Dependency to get the current authenticated user.

    Usage:
        @app.get("/protected")
        async def protected_route(user: TokenData = Depends(get_current_user)):
            return {"user_id": user.user_id}
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials
    token_data = verify_token(token)

    if token_data is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Verify session is still active
    if token_data.session_id:
        session = db.get_session(token_data.session_id)
        if not session or not session.get("is_active"):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Session has been invalidated",
                headers={"WWW-Authenticate": "Bearer"},
            )

    # Verify user still exists and is active
    user = db.get_user_by_id(token_data.user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )

    if user["account_status"] == "BLOCKED":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account has been blocked",
        )

    if user["account_status"] == "SUSPENDED":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account has been suspended",
        )

    return token_data


async def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[TokenData]:
    """
    Dependency to optionally get the current user.
    Returns None if not authenticated (doesn't raise exception).
    """
    if credentials is None:
        return None

    token = credentials.credentials
    return verify_token(token)


def require_roles(allowed_roles: List[str]):
    """
    Dependency factory for role-based access control.

    Usage:
        @app.get("/admin-only")
        async def admin_route(user: TokenData = Depends(require_roles(["ADMIN"]))):
            return {"message": "Admin access granted"}
    """
    async def role_checker(user: TokenData = Depends(get_current_user)) -> TokenData:
        if not can_access_route(user.user_type, allowed_roles):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied. Required roles: {', '.join(allowed_roles)}",
            )
        return user

    return role_checker


def get_client_info(request: Request) -> dict:
    """
    Extract client information from request for device tracking.
    """
    # Get IP address (handle proxies)
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        ip = forwarded.split(",")[0].strip()
    else:
        ip = request.client.host if request.client else "unknown"

    # Get user agent
    user_agent = request.headers.get("User-Agent", "unknown")

    # Get or generate device ID from cookie/header
    device_id = request.headers.get("X-Device-ID")
    if not device_id:
        device_id = request.cookies.get("device_id")
    if not device_id:
        device_id = f"device_{uuid.uuid4().hex[:12]}"

    return {
        "ip_address": ip,
        "user_agent": user_agent,
        "device_id": device_id,
    }


# Convenience dependencies for common role patterns
require_customer = require_roles(["CUSTOMER", "ADMIN"])
require_merchant = require_roles(["MERCHANT", "ADMIN"])
require_admin = require_roles(["ADMIN"])
require_customer_or_merchant = require_roles(["CUSTOMER", "MERCHANT", "ADMIN"])
