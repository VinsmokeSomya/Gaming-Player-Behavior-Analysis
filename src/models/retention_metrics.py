"""
Retention Metrics data model with validation.
"""
from dataclasses import dataclass
from datetime import date
from typing import Optional


@dataclass
class RetentionMetrics:
    """Data model for retention metrics by cohort."""
    
    cohort_date: date
    day_1_retention: float
    day_7_retention: float
    day_30_retention: float
    cohort_size: int
    segment: str
    
    def __post_init__(self):
        """Validate data after initialization."""
        self._validate_cohort_date()
        self._validate_retention_rates()
        self._validate_cohort_size()
        self._validate_segment()
    
    def _validate_cohort_date(self):
        """Validate cohort date."""
        if not isinstance(self.cohort_date, date):
            raise ValueError("cohort_date must be a date object")
    
    def _validate_retention_rates(self):
        """Validate retention rate values."""
        retention_fields = [
            ('day_1_retention', self.day_1_retention),
            ('day_7_retention', self.day_7_retention),
            ('day_30_retention', self.day_30_retention)
        ]
        
        for field_name, value in retention_fields:
            if not isinstance(value, (int, float)):
                raise ValueError(f"{field_name} must be a number")
            
            if not 0 <= value <= 1:
                raise ValueError(f"{field_name} must be between 0 and 1")
        
        # Logical validation: retention should generally decrease over time
        if self.day_7_retention > self.day_1_retention:
            raise ValueError("day_7_retention should not exceed day_1_retention")
        
        if self.day_30_retention > self.day_7_retention:
            raise ValueError("day_30_retention should not exceed day_7_retention")
    
    def _validate_cohort_size(self):
        """Validate cohort size."""
        if not isinstance(self.cohort_size, int) or self.cohort_size <= 0:
            raise ValueError("cohort_size must be a positive integer")
    
    def _validate_segment(self):
        """Validate segment field."""
        if not self.segment or not isinstance(self.segment, str):
            raise ValueError("segment must be a non-empty string")
        
        # Validate segment format (alphanumeric with underscores and hyphens)
        import re
        if not re.match(r'^[a-zA-Z0-9_-]+$', self.segment):
            raise ValueError("segment must contain only alphanumeric characters, underscores, and hyphens")
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'cohort_date': self.cohort_date.isoformat(),
            'day_1_retention': self.day_1_retention,
            'day_7_retention': self.day_7_retention,
            'day_30_retention': self.day_30_retention,
            'cohort_size': self.cohort_size,
            'segment': self.segment
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'RetentionMetrics':
        """Create instance from dictionary."""
        return cls(
            cohort_date=date.fromisoformat(data['cohort_date']),
            day_1_retention=data['day_1_retention'],
            day_7_retention=data['day_7_retention'],
            day_30_retention=data['day_30_retention'],
            cohort_size=data['cohort_size'],
            segment=data['segment']
        )
    
    def get_retention_at_day(self, day: int) -> Optional[float]:
        """Get retention rate for a specific day."""
        retention_map = {
            1: self.day_1_retention,
            7: self.day_7_retention,
            30: self.day_30_retention
        }
        return retention_map.get(day)