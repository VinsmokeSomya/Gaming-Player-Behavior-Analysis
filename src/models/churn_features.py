"""
Churn Features data model with validation.
"""
from dataclasses import dataclass, asdict
from datetime import date
from typing import Dict, Any
import re


@dataclass
class ChurnFeatures:
    """Data model for churn prediction features."""
    
    player_id: str
    days_since_last_session: int
    sessions_last_7_days: int
    avg_session_duration_minutes: float
    levels_completed_last_week: int
    purchases_last_30_days: float
    social_connections: int
    feature_date: date
    
    def __post_init__(self):
        """Validate data after initialization."""
        self._validate_player_id()
        self._validate_numeric_fields()
        self._validate_feature_date()
    
    def _validate_player_id(self):
        """Validate player ID format."""
        if not self.player_id or not isinstance(self.player_id, str):
            raise ValueError("player_id must be a non-empty string")
        
        if not re.match(r'^[a-zA-Z0-9_-]+$', self.player_id):
            raise ValueError("player_id must contain only alphanumeric characters, underscores, and hyphens")
    
    def _validate_numeric_fields(self):
        """Validate all numeric fields."""
        # Days since last session
        if not isinstance(self.days_since_last_session, int) or self.days_since_last_session < 0:
            raise ValueError("days_since_last_session must be a non-negative integer")
        
        # Sessions last 7 days
        if not isinstance(self.sessions_last_7_days, int) or self.sessions_last_7_days < 0:
            raise ValueError("sessions_last_7_days must be a non-negative integer")
        
        # Average session duration
        if not isinstance(self.avg_session_duration_minutes, (int, float)) or self.avg_session_duration_minutes < 0:
            raise ValueError("avg_session_duration_minutes must be a non-negative number")
        
        # Levels completed
        if not isinstance(self.levels_completed_last_week, int) or self.levels_completed_last_week < 0:
            raise ValueError("levels_completed_last_week must be a non-negative integer")
        
        # Purchases
        if not isinstance(self.purchases_last_30_days, (int, float)) or self.purchases_last_30_days < 0:
            raise ValueError("purchases_last_30_days must be a non-negative number")
        
        # Social connections
        if not isinstance(self.social_connections, int) or self.social_connections < 0:
            raise ValueError("social_connections must be a non-negative integer")
        
        # Business logic validations
        if self.sessions_last_7_days > 0 and self.days_since_last_session > 7:
            raise ValueError("Cannot have sessions in last 7 days if last session was more than 7 days ago")
    
    def _validate_feature_date(self):
        """Validate feature date."""
        if not isinstance(self.feature_date, date):
            raise ValueError("feature_date must be a date object")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        # Convert date to ISO string
        data['feature_date'] = self.feature_date.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChurnFeatures':
        """Create instance from dictionary."""
        # Convert ISO string to date if needed
        if isinstance(data['feature_date'], str):
            data['feature_date'] = date.fromisoformat(data['feature_date'])
        
        return cls(**data)
    
    def is_high_churn_risk(self) -> bool:
        """Determine if player shows high churn risk based on features."""
        # Simple heuristic for high churn risk
        risk_factors = 0
        
        if self.days_since_last_session > 3:
            risk_factors += 1
        
        if self.sessions_last_7_days == 0:
            risk_factors += 2
        
        if self.avg_session_duration_minutes < 5:
            risk_factors += 1
        
        if self.levels_completed_last_week == 0:
            risk_factors += 1
        
        if self.purchases_last_30_days == 0:
            risk_factors += 1
        
        return risk_factors >= 3
    
    def calculate_engagement_score(self) -> float:
        """Calculate engagement score from 0 to 1."""
        # Normalize and combine different engagement factors
        session_score = min(self.sessions_last_7_days / 7, 1.0)  # Max 1 session per day
        duration_score = min(self.avg_session_duration_minutes / 30, 1.0)  # Max 30 min sessions
        progression_score = min(self.levels_completed_last_week / 10, 1.0)  # Max 10 levels per week
        
        # Weight the scores
        engagement_score = (session_score * 0.4 + duration_score * 0.3 + progression_score * 0.3)
        
        # Penalize for inactivity
        if self.days_since_last_session > 0:
            inactivity_penalty = min(self.days_since_last_session / 7, 1.0)
            engagement_score *= (1 - inactivity_penalty)
        
        return max(0.0, min(1.0, engagement_score))