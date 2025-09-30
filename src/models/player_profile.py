"""
Player Profile data model with validation.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import re


@dataclass
class PlayerProfile:
    """Data model for player profile information."""
    
    player_id: str
    registration_date: datetime
    last_active_date: datetime
    total_sessions: int
    total_playtime_minutes: int
    highest_level_reached: int
    total_purchases: float
    churn_risk_score: float
    churn_prediction_date: datetime
    
    def __post_init__(self):
        """Validate data after initialization."""
        self._validate_player_id()
        self._validate_dates()
        self._validate_numeric_fields()
        self._validate_churn_risk_score()
    
    def _validate_player_id(self):
        """Validate player ID format."""
        if not self.player_id or not isinstance(self.player_id, str):
            raise ValueError("player_id must be a non-empty string")
        
        if not re.match(r'^[a-zA-Z0-9_-]+$', self.player_id):
            raise ValueError("player_id must contain only alphanumeric characters, underscores, and hyphens")
    
    def _validate_dates(self):
        """Validate date fields."""
        if not isinstance(self.registration_date, datetime):
            raise ValueError("registration_date must be a datetime object")
        
        if not isinstance(self.last_active_date, datetime):
            raise ValueError("last_active_date must be a datetime object")
        
        if not isinstance(self.churn_prediction_date, datetime):
            raise ValueError("churn_prediction_date must be a datetime object")
        
        if self.last_active_date < self.registration_date:
            raise ValueError("last_active_date cannot be before registration_date")
    
    def _validate_numeric_fields(self):
        """Validate numeric fields."""
        if not isinstance(self.total_sessions, int) or self.total_sessions < 0:
            raise ValueError("total_sessions must be a non-negative integer")
        
        if not isinstance(self.total_playtime_minutes, int) or self.total_playtime_minutes < 0:
            raise ValueError("total_playtime_minutes must be a non-negative integer")
        
        if not isinstance(self.highest_level_reached, int) or self.highest_level_reached < 0:
            raise ValueError("highest_level_reached must be a non-negative integer")
        
        if not isinstance(self.total_purchases, (int, float)) or self.total_purchases < 0:
            raise ValueError("total_purchases must be a non-negative number")
    
    def _validate_churn_risk_score(self):
        """Validate churn risk score is between 0 and 1."""
        if not isinstance(self.churn_risk_score, (int, float)):
            raise ValueError("churn_risk_score must be a number")
        
        if not 0 <= self.churn_risk_score <= 1:
            raise ValueError("churn_risk_score must be between 0 and 1")
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'player_id': self.player_id,
            'registration_date': self.registration_date.isoformat(),
            'last_active_date': self.last_active_date.isoformat(),
            'total_sessions': self.total_sessions,
            'total_playtime_minutes': self.total_playtime_minutes,
            'highest_level_reached': self.highest_level_reached,
            'total_purchases': self.total_purchases,
            'churn_risk_score': self.churn_risk_score,
            'churn_prediction_date': self.churn_prediction_date.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'PlayerProfile':
        """Create instance from dictionary."""
        return cls(
            player_id=data['player_id'],
            registration_date=datetime.fromisoformat(data['registration_date']),
            last_active_date=datetime.fromisoformat(data['last_active_date']),
            total_sessions=data['total_sessions'],
            total_playtime_minutes=data['total_playtime_minutes'],
            highest_level_reached=data['highest_level_reached'],
            total_purchases=data['total_purchases'],
            churn_risk_score=data['churn_risk_score'],
            churn_prediction_date=datetime.fromisoformat(data['churn_prediction_date'])
        )