"""Database models for user authentication and authorization."""
from datetime import datetime
from typing import List, Optional
from sqlalchemy import (
    Column, String, Integer, DateTime, ForeignKey, Table, Boolean, Text
)
from sqlalchemy.orm import relationship, declarative_base
from sqlalchemy.dialects.postgresql import UUID
import uuid

Base = declarative_base()

# Association table for many-to-many user_roles
user_roles_table = Table(
    "user_roles",
    Base.metadata,
    Column("user_id", UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), primary_key=True),
    Column("role_id", UUID(as_uuid=True), ForeignKey("roles.id", ondelete="CASCADE"), primary_key=True),
)


class User(Base):
    """User model with IAP identity."""
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False, index=True)
    iap_subject = Column(String(255), unique=True, nullable=True, index=True)  # X-Goog-Authenticated-User-Id
    display_name = Column(String(255), nullable=True)
    status = Column(String(50), default="active")  # active, suspended, pending
    tier_id = Column(UUID(as_uuid=True), ForeignKey("rate_limit_tiers.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login_at = Column(DateTime, nullable=True)
    
    # Relationships
    roles = relationship("Role", secondary=user_roles_table, back_populates="users")
    tier = relationship("RateLimitTier", back_populates="users")
    
    def has_role(self, role_name: str) -> bool:
        """Check if user has a specific role."""
        return any(role.name == role_name for role in self.roles)
    
    def has_any_role(self, role_names: List[str]) -> bool:
        """Check if user has any of the specified roles."""
        return any(role.name in role_names for role in self.roles)
    
    def is_active(self) -> bool:
        """Check if user account is active."""
        return self.status == "active"
    
    def to_dict(self):
        """Serialize user to dictionary."""
        return {
            "id": str(self.id),
            "email": self.email,
            "display_name": self.display_name,
            "status": self.status,
            "tier": self.tier.name if self.tier else None,
            "roles": [role.name for role in self.roles],
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_login_at": self.last_login_at.isoformat() if self.last_login_at else None,
        }


class Role(Base):
    """Role model for RBAC."""
    __tablename__ = "roles"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(50), unique=True, nullable=False, index=True)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    users = relationship("User", secondary=user_roles_table, back_populates="roles")


class RateLimitTier(Base):
    """Rate limit tier configuration."""
    __tablename__ = "rate_limit_tiers"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(50), unique=True, nullable=False, index=True)
    requests_per_minute = Column(Integer, default=0)
    requests_per_hour = Column(Integer, default=0)
    requests_per_day = Column(Integer, default=0)
    description = Column(Text, nullable=True)
    is_default = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    users = relationship("User", back_populates="tier")
    
    def to_dict(self):
        return {
            "id": str(self.id),
            "name": self.name,
            "requests_per_minute": self.requests_per_minute,
            "requests_per_hour": self.requests_per_hour,
            "requests_per_day": self.requests_per_day,
            "description": self.description,
        }
