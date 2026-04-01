"""User registration and management endpoints."""
from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session

from src.database.pg_connection import get_db
from src.auth.dependencies import (
    get_current_user, get_current_user_optional, get_iap_email, 
    get_iap_subject, require_admin
)
from src.models import User, Role, RateLimitTier

router = APIRouter(prefix="/auth", tags=["authentication"])


# Pydantic schemas
class UserRegistration(BaseModel):
    display_name: Optional[str] = None


class UserResponse(BaseModel):
    id: str
    email: str
    display_name: Optional[str]
    status: str
    tier: Optional[str]
    roles: List[str]
    created_at: Optional[str]
    last_login_at: Optional[str]
    
    class Config:
        from_attributes = True


def user_to_response(user) -> UserResponse:
    """Convert User model to UserResponse dict."""
    return {
        "id": str(user.id),
        "email": user.email,
        "display_name": user.display_name,
        "status": user.status,
        "tier": user.tier.name if user.tier else None,
        "roles": [role.name for role in user.roles],
        "created_at": user.created_at.isoformat() if user.created_at else None,
        "last_login_at": user.last_login_at.isoformat() if user.last_login_at else None,
    }


class RoleResponse(BaseModel):
    id: str
    name: str
    description: Optional[str]


class TierResponse(BaseModel):
    id: str
    name: str
    requests_per_minute: int
    requests_per_hour: int
    requests_per_day: int
    description: Optional[str]


@router.post("/register")
async def register_user(
    request: Request,
    registration: UserRegistration,
    db: Session = Depends(get_db)
):
    """Register a new user from IAP identity."""
    email = get_iap_email(request)
    subject = get_iap_subject(request)
    
    if not email:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="IAP authentication required. No user email found in headers.",
        )
    
    # Check if user already exists
    existing_user = db.query(User).filter(User.email == email).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="User already registered.",
        )
    
    # Get default tier (blocked/lowest)
    default_tier = db.query(RateLimitTier).filter(RateLimitTier.is_default == True).first()
    if not default_tier:
        # Create default blocked tier if not exists
        default_tier = RateLimitTier(
            name="blocked",
            requests_per_minute=0,
            requests_per_hour=0,
            requests_per_day=0,
            description="Blocked tier - no API access",
            is_default=True,
        )
        db.add(default_tier)
        db.commit()
    
    # Get or create default user role
    user_role = db.query(Role).filter(Role.name == "user").first()
    if not user_role:
        user_role = Role(name="user", description="Standard user")
        db.add(user_role)
        db.commit()
    
    # Create new user
    new_user = User(
        email=email,
        iap_subject=subject,
        display_name=registration.display_name or email.split("@")[0],
        status="active",
        tier_id=default_tier.id,
        last_login_at=datetime.utcnow(),
    )
    new_user.roles.append(user_role)
    
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    return user_to_response(new_user)


@router.get("/me")
async def get_me(current_user: User = Depends(get_current_user)):
    """Get current user profile."""
    return user_to_response(current_user)


@router.get("/check")
async def check_auth(request: Request, db: Session = Depends(get_db)):
    """Check if user is authenticated and registered."""
    email = get_iap_email(request)
    if not email:
        return {
            "authenticated": False,
            "registered": False,
            "message": "Not authenticated via IAP",
        }
    
    user = db.query(User).filter(User.email == email).first()
    if not user:
        return {
            "authenticated": True,
            "registered": False,
            "email": email,
            "message": "Authenticated but not registered. Please complete registration.",
        }
    
    return {
        "authenticated": True,
        "registered": True,
        "user": user.to_dict(),
    }


# Admin endpoints
@router.get("/admin/users")
async def list_users(
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
    skip: int = 0,
    limit: int = 100,
):
    """List all users (admin only)."""
    users = db.query(User).offset(skip).limit(limit).all()
    return [user_to_response(u) for u in users]


@router.post("/admin/users/{user_id}/roles")
async def assign_role(
    user_id: str,
    role_name: str,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Assign a role to a user (admin only)."""
    from uuid import UUID
    
    user = db.query(User).filter(User.id == UUID(user_id)).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )
    
    role = db.query(Role).filter(Role.name == role_name).first()
    if not role:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Role '{role_name}' not found",
        )
    
    if role not in user.roles:
        user.roles.append(role)
        db.commit()
        db.refresh(user)
    
    return user_to_response(user)


@router.post("/admin/users/{user_id}/tier")
async def set_user_tier(
    user_id: str,
    tier_name: str,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Set user's rate limit tier (admin only)."""
    from uuid import UUID
    
    user = db.query(User).filter(User.id == UUID(user_id)).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )
    
    tier = db.query(RateLimitTier).filter(RateLimitTier.name == tier_name).first()
    if not tier:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tier '{tier_name}' not found",
        )
    
    user.tier_id = tier.id
    db.commit()
    db.refresh(user)
    
    return user_to_response(user)


@router.post("/admin/users/{user_id}/suspend")
async def suspend_user(
    user_id: str,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Suspend a user account (admin only)."""
    from uuid import UUID
    
    user = db.query(User).filter(User.id == UUID(user_id)).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )
    
    user.status = "suspended"
    db.commit()
    
    return {"message": f"User {user.email} suspended"}


@router.post("/admin/tiers")
async def create_tier(
    name: str,
    requests_per_minute: int,
    requests_per_hour: int = 0,
    requests_per_day: int = 0,
    description: Optional[str] = None,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Create a new rate limit tier (admin only)."""
    existing = db.query(RateLimitTier).filter(RateLimitTier.name == name).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Tier '{name}' already exists",
        )
    
    tier = RateLimitTier(
        name=name,
        requests_per_minute=requests_per_minute,
        requests_per_hour=requests_per_hour,
        requests_per_day=requests_per_day,
        description=description,
    )
    db.add(tier)
    db.commit()
    db.refresh(tier)
    
    return tier.to_dict()
