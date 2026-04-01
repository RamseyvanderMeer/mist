"""Script to create admin user for Ramsey."""
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from sqlalchemy.orm import Session
from src.database.pg_connection import pg_engine, PGSessionLocal
from src.database.init import init_db
from src.models import User, Role, RateLimitTier


def create_admin_user(email: str, display_name: str = None):
    """Create or upgrade user to admin."""
    db = PGSessionLocal()
    try:
        # Initialize DB with default roles/tiers
        init_db(db)
        
        # Find or create user
        user = db.query(User).filter(User.email == email.lower()).first()
        
        if not user:
            # Get admin tier
            admin_tier = db.query(RateLimitTier).filter(RateLimitTier.name == "admin").first()
            if not admin_tier:
                print("Error: Admin tier not found")
                return
            
            # Create user
            user = User(
                email=email.lower(),
                display_name=display_name or email.split("@")[0],
                status="active",
                tier_id=admin_tier.id,
            )
            db.add(user)
            db.commit()
            db.refresh(user)
            print(f"Created user: {email}")
        else:
            print(f"User exists: {email}")
        
        # Assign admin role
        admin_role = db.query(Role).filter(Role.name == "admin").first()
        if not admin_role:
            print("Error: Admin role not found")
            return
        
        if admin_role not in user.roles:
            user.roles.append(admin_role)
            print(f"Added admin role to {email}")
        else:
            print(f"User already has admin role")
        
        # Set admin tier
        admin_tier = db.query(RateLimitTier).filter(RateLimitTier.name == "admin").first()
        if user.tier_id != admin_tier.id:
            user.tier_id = admin_tier.id
            print(f"Upgraded {email} to admin tier")
        else:
            print(f"User already on admin tier")
        
        db.commit()
        print(f"\n✅ {email} is now an admin with full access!")
        print(f"   Tier: {user.tier.name}")
        print(f"   Roles: {[r.name for r in user.roles]}")
        
    finally:
        db.close()


if __name__ == "__main__":
    # Admin emails
    ADMIN_EMAILS = [
        ("ramsey12223@gmail.com", "Ramsey"),
        ("ramvandermeer@gmail.com", "Ramsey Alt"),
    ]
    
    for email, name in ADMIN_EMAILS:
        create_admin_user(email, display_name=name)
        print("-" * 50)
