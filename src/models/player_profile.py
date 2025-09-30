"""
Player profile data model for retention analytics.
"""

from dataclasses import dataclass, asdict
from datetime import datetime, date
from typing import Optional, Dict, Any


@dataclass
class PlayerProfile:
    """Represents a player's profile with key metrics for retention analysis."""
    
    player_id: str
    registration_date: datetime
    last_active_date: datetime
    total_sessions: int
    total_playtime_minutes: int
    highest_level_reached: int
    total_purchases: float
    churn_risk_score: float
    churn_prediction_date: datetime
    
    # Optional demographic fields
    age: Optional[int] = None
    gender: Optional[str] = None
    location: Optional[str] = None
    device_type: Optional[str] = None
    
    # Engagement metrics
    avg_session_duration: Optional[float] = None
    sessions_last_7_days: Optional[int] = None
    sessions_last_30_days: Optional[int] = None
    days_since_last_session: Optional[int] = None
    
    # Monetization metrics
    first_purchase_date: Optional[datetime] = None
    last_purchase_date: Optional[datetime] = None
    purchase_frequency: Optional[float] = None
    avg_purchase_amount: Optional[float] = None
    
    def __post_init__(self):
        """Validate data after initialization."""
        if self.churn_risk_score < 0 or self.churn_risk_score > 1:
            raise ValueError(f"Churn risk score must be between 0 and 1, got {self.churn_risk_score}")
        
        if self.total_sessions < 0:
            raise ValueError(f"Total sessions cannot be negative, got {self.total_sessions}")
        
        if self.total_playtime_minutes < 0:
            raise ValueError(f"Total playtime cannot be negative, got {self.total_playtime_minutes}")
        
        if self.registration_date > self.last_active_date:
            raise ValueError("Registration date cannot be after last active date")
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PlayerProfile':
        """Create PlayerProfile from dictionary data."""
        # Convert string dates to datetime objects if needed
        for field in ['registration_date', 'last_active_date', 'churn_prediction_date', 
                     'first_purchase_date', 'last_purchase_date']:
            if field in data and isinstance(data[field], str):
                data[field] = datetime.fromisoformat(data[field])
        
        return cls(**data)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert PlayerProfile to dictionary."""
        data = asdict(self)
        
        # Convert datetime objects to ISO strings for JSON serialization
        for field in ['registration_date', 'last_active_date', 'churn_prediction_date',
                     'first_purchase_date', 'last_purchase_date']:
            if data[field] is not None:
                data[field] = data[field].isoformat()
        
        return data
    
    def calculate_lifetime_value(self) -> float:
        """Calculate estimated lifetime value based on current spending patterns."""
        if self.total_purchases <= 0:
            return 0.0
        
        # Simple LTV calculation: current spending / churn risk
        # Lower churn risk = higher expected lifetime
        risk_factor = max(0.1, self.churn_risk_score)  # Avoid division by zero
        return self.total_purchases / risk_factor
    
    def get_engagement_level(self) -> str:
        """Categorize player engagement level."""
        if self.total_sessions >= 100 and self.total_purchases > 50:
            return 'premium'
        elif self.total_sessions >= 50:
            return 'core'
        elif self.total_sessions >= 10:
            return 'casual'
        else:
            return 'new'
    
    def get_churn_risk_category(self) -> str:
        """Categorize churn risk level."""
        if self.churn_risk_score >= 0.7:
            return 'high'
        elif self.churn_risk_score >= 0.3:
            return 'medium'
        else:
            return 'low'
    
    def days_since_registration(self) -> int:
        """Calculate days since player registration."""
        return (datetime.now() - self.registration_date).days
    
    def is_active_player(self, days_threshold: int = 7) -> bool:
        """Check if player is considered active based on last session."""
        days_inactive = (datetime.now() - self.last_active_date).days
        return days_inactive <= days_threshold
    
    def __str__(self) -> str:
        """String representation of player profile."""
        return (f"PlayerProfile(id={self.player_id}, "
                f"sessions={self.total_sessions}, "
                f"churn_risk={self.churn_risk_score:.3f}, "
                f"engagement={self.get_engagement_level()})")