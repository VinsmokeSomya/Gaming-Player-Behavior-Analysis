"""
Validation functions for player data and analytics models.
"""

from datetime import datetime, date
from typing import Dict, Any, List, Optional
import re


def validate_player_event(event: Dict[str, Any]) -> bool:
    """Validate a player event dictionary."""
    required_fields = ['player_id', 'event_type', 'timestamp']
    
    # Check required fields
    for field in required_fields:
        if field not in event:
            raise ValueError(f"Missing required field: {field}")
    
    # Validate player_id format
    if not isinstance(event['player_id'], str) or not event['player_id'].strip():
        raise ValueError("player_id must be a non-empty string")
    
    # Validate event_type
    valid_event_types = [
        'session_start', 'session_end', 'level_complete', 
        'purchase', 'achievement_unlock', 'tutorial_complete'
    ]
    if event['event_type'] not in valid_event_types:
        raise ValueError(f"Invalid event_type: {event['event_type']}. Must be one of {valid_event_types}")
    
    # Validate timestamp
    if not isinstance(event['timestamp'], (datetime, str)):
        raise ValueError("timestamp must be datetime object or ISO string")
    
    # Validate event-specific fields
    if event['event_type'] == 'purchase':
        if 'purchase_amount' not in event:
            raise ValueError("purchase events must have purchase_amount field")
        if not isinstance(event['purchase_amount'], (int, float)) or event['purchase_amount'] < 0:
            raise ValueError("purchase_amount must be a non-negative number")
    
    if event['event_type'] == 'level_complete':
        if 'level' not in event:
            raise ValueError("level_complete events must have level field")
        if not isinstance(event['level'], int) or event['level'] < 1:
            raise ValueError("level must be a positive integer")
    
    if event['event_type'] in ['session_start', 'session_end']:
        if 'session_duration' in event:
            if not isinstance(event['session_duration'], (int, float)) or event['session_duration'] < 0:
                raise ValueError("session_duration must be a non-negative number")
    
    return True


def validate_retention_calculation(cohort_data: List[Dict[str, Any]]) -> bool:
    """Validate cohort retention calculation data."""
    if not cohort_data:
        raise ValueError("Cohort data cannot be empty")
    
    required_fields = ['cohort_date', 'period', 'retention_rate', 'cohort_size']
    
    for i, record in enumerate(cohort_data):
        for field in required_fields:
            if field not in record:
                raise ValueError(f"Record {i}: Missing required field {field}")
        
        # Validate retention_rate
        if not isinstance(record['retention_rate'], (int, float)):
            raise ValueError(f"Record {i}: retention_rate must be numeric")
        if record['retention_rate'] < 0 or record['retention_rate'] > 1:
            raise ValueError(f"Record {i}: retention_rate must be between 0 and 1")
        
        # Validate cohort_size
        if not isinstance(record['cohort_size'], int) or record['cohort_size'] < 0:
            raise ValueError(f"Record {i}: cohort_size must be a non-negative integer")
        
        # Validate period
        if not isinstance(record['period'], int) or record['period'] < 0:
            raise ValueError(f"Record {i}: period must be a non-negative integer")
    
    return True


def validate_metric_thresholds(metrics: Dict[str, float]) -> bool:
    """Validate analytics metric thresholds."""
    valid_metrics = {
        'churn_risk_threshold': (0.0, 1.0),
        'engagement_threshold': (0.0, float('inf')),
        'revenue_threshold': (0.0, float('inf')),
        'session_threshold': (0, float('inf'))
    }
    
    for metric_name, value in metrics.items():
        if metric_name not in valid_metrics:
            raise ValueError(f"Unknown metric: {metric_name}")
        
        min_val, max_val = valid_metrics[metric_name]
        if not isinstance(value, (int, float)):
            raise ValueError(f"{metric_name} must be numeric")
        if value < min_val or value > max_val:
            raise ValueError(f"{metric_name} must be between {min_val} and {max_val}")
    
    return True


def validate_date_range(start_date: date, end_date: date) -> bool:
    """Validate date range for analytics queries."""
    if not isinstance(start_date, date):
        raise ValueError("start_date must be a date object")
    if not isinstance(end_date, date):
        raise ValueError("end_date must be a date object")
    
    if start_date > end_date:
        raise ValueError("start_date cannot be after end_date")
    
    # Check for reasonable date range (not too far in past/future)
    today = date.today()
    max_past_days = 365 * 3  # 3 years
    max_future_days = 30     # 30 days
    
    if (today - start_date).days > max_past_days:
        raise ValueError(f"start_date cannot be more than {max_past_days} days in the past")
    
    if (end_date - today).days > max_future_days:
        raise ValueError(f"end_date cannot be more than {max_future_days} days in the future")
    
    return True


def sanitize_player_segment(segment: str) -> str:
    """Sanitize and validate player segment string."""
    if not isinstance(segment, str):
        raise ValueError("Player segment must be a string")
    
    # Clean the segment string
    segment = segment.strip().lower()
    
    # Validate against allowed segments
    valid_segments = ['new', 'casual', 'core', 'premium', 'churned', 'returning']
    
    if segment not in valid_segments:
        # Try to map common variations
        segment_mapping = {
            'beginner': 'new',
            'starter': 'new',
            'regular': 'casual',
            'active': 'core',
            'engaged': 'core',
            'vip': 'premium',
            'whale': 'premium',
            'inactive': 'churned',
            'lapsed': 'churned'
        }
        
        if segment in segment_mapping:
            segment = segment_mapping[segment]
        else:
            raise ValueError(f"Invalid player segment: {segment}. Must be one of {valid_segments}")
    
    return segment


def validate_player_id(player_id: str) -> bool:
    """Validate player ID format."""
    if not isinstance(player_id, str):
        raise ValueError("Player ID must be a string")
    
    if not player_id.strip():
        raise ValueError("Player ID cannot be empty")
    
    # Check for reasonable length
    if len(player_id) > 100:
        raise ValueError("Player ID too long (max 100 characters)")
    
    # Check for valid characters (alphanumeric, underscore, hyphen)
    if not re.match(r'^[a-zA-Z0-9_-]+$', player_id):
        raise ValueError("Player ID can only contain letters, numbers, underscores, and hyphens")
    
    return True


def validate_churn_features(features: Dict[str, Any]) -> bool:
    """Validate churn prediction features."""
    required_fields = [
        'player_id', 'days_since_last_session', 'sessions_last_7_days',
        'avg_session_duration_minutes', 'levels_completed_last_week',
        'purchases_last_30_days', 'social_connections'
    ]
    
    for field in required_fields:
        if field not in features:
            raise ValueError(f"Missing required churn feature: {field}")
    
    # Validate numeric fields
    numeric_fields = {
        'days_since_last_session': (0, 365),
        'sessions_last_7_days': (0, 1000),
        'avg_session_duration_minutes': (0, 1440),  # Max 24 hours
        'levels_completed_last_week': (0, 1000),
        'purchases_last_30_days': (0, 10000),
        'social_connections': (0, 10000)
    }
    
    for field, (min_val, max_val) in numeric_fields.items():
        value = features[field]
        if not isinstance(value, (int, float)):
            raise ValueError(f"{field} must be numeric")
        if value < min_val or value > max_val:
            raise ValueError(f"{field} must be between {min_val} and {max_val}")
    
    # Validate player_id
    validate_player_id(features['player_id'])
    
    return True