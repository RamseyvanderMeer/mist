"""Database initialization and seeding."""
from sqlalchemy.orm import Session
from src.database.pg_connection import PGSessionLocal
from src.models import Role, RateLimitTier


def init_db(db: Session):
    """Initialize database with default roles and tiers."""
    _init_roles(db)
    _init_tiers(db)
    db.commit()


def _init_roles(db: Session):
    """Create default roles if they don't exist."""
    default_roles = [
        {"name": "user", "description": "Standard user with basic access"},
        {"name": "admin", "description": "Administrator with full access"},
        {"name": "beta", "description": "Beta tester with early access"},
        {"name": "internal_service", "description": "Internal service account"},
    ]
    
    for role_data in default_roles:
        existing = db.query(Role).filter(Role.name == role_data["name"]).first()
        if not existing:
            role = Role(**role_data)
            db.add(role)
    
    db.commit()


def _init_tiers(db: Session):
    """Create default rate limit tiers if they don't exist."""
    default_tiers = [
        {
            "name": "blocked",
            "requests_per_minute": 0,
            "requests_per_hour": 0,
            "requests_per_day": 0,
            "description": "Blocked - no API access (default for new users)",
            "is_default": True,
        },
        {
            "name": "free",
            "requests_per_minute": 10,
            "requests_per_hour": 100,
            "requests_per_day": 500,
            "description": "Free tier with basic limits",
            "is_default": False,
        },
        {
            "name": "premium",
            "requests_per_minute": 100,
            "requests_per_hour": 1000,
            "requests_per_day": 5000,
            "description": "Premium tier with higher limits",
            "is_default": False,
        },
        {
            "name": "admin",
            "requests_per_minute": 1000,
            "requests_per_hour": 10000,
            "requests_per_day": 100000,
            "description": "Admin tier with unlimited access",
            "is_default": False,
        },
    ]
    
    for tier_data in default_tiers:
        existing = db.query(RateLimitTier).filter(RateLimitTier.name == tier_data["name"]).first()
        if not existing:
            tier = RateLimitTier(**tier_data)
            db.add(tier)
    
    db.commit()
