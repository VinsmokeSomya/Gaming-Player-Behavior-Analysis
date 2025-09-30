"""
Churn Features data model with validation.
"""
from dataclasses import dataclass
from datetime import date
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
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'player_id': self.player_id,
            'days_since_last_session': self.days_since_last_session,
            'sessions_last_7_days': self.sessions_last_7_days,
            'avg_session_duration_minutes': self.avg_session_duration_minutes,
            'levels_completed_last_week': self.levels_completed_last_week,
            'purchases_last_30_days': self.purchases_last_30_days,
            'social_connections': self.social_connections,
            'feature_date': self.feature_date.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ChurnFeatures':
        """Create instance from dictionary."""
        return cls(
            player_id=data['player_id'],
            days_since_last_session=data['days_since_last_session'],
            sessions_last_7_days=data['sessions_last_7_days'],
            avg_session_duration_minutes=data['avg_session_duration_minutes'],
            levels_completed_last_week=data['levels_completed_last_week'],
            purchases_last_30_days=data['purchases_last_30_days'],
            social_connections=data['social_connections'],
            feature_date=date.fromisoformat(data['feature_date'])
        )
    
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