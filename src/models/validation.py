"""
Data validation functions for player events and metrics.
"""
from datetime import datetime, date
from typing import Dict, Any, List, Optional
import re


def validate_player_event(event_data: Dict[str, Any]) -> bool:
    """
    Validate player event data structure and values.
    
    Args:
        event_data: Dictionary containing player event data
        
    Returns:
        bool: True if valid, raises ValueError if invalid
        
    Raises:
        ValueError: If validation fails
    """
    required_fields = ['player_id', 'event_type', 'timestamp']
    
    # Check required fields
    for field in required_fields:
        if field not in event_data:
            raise ValueError(f"Missing required field: {field}")
    
    # Validate player_id
    player_id = event_data['player_id']
    if not isinstance(player_id, str) or not player_id:
        raise ValueError("player_id must be a non-empty string")
    
    if not re.match(r'^[a-zA-Z0-9_-]+$', player_id):
        raise ValueError("player_id must contain only alphanumeric characters, underscores, and hyphens")
    
    # Validate event_type
    event_type = event_data['event_type']
    valid_event_types = [
        'session_start', 'session_end', 'level_complete', 'level_fail',
        'purchase', 'achievement_unlock', 'tutorial_complete', 'app_install'
    ]
    
    if not isinstance(event_type, str) or event_type not in valid_event_types:
        raise ValueError(f"event_type must be one of: {', '.join(valid_event_types)}")
    
    # Validate timestamp
    timestamp = event_data['timestamp']
    if isinstance(timestamp, str):
        try:
            datetime.fromisoformat(timestamp)
        except ValueError:
            raise ValueError("timestamp must be a valid ISO format datetime string")
    elif not isinstance(timestamp, datetime):
        raise ValueError("timestamp must be a datetime object or ISO format string")
    
    # Validate optional fields if present
    if 'level' in event_data:
        level = event_data['level']
        if not isinstance(level, int) or level < 1:
            raise ValueError("level must be a positive integer")
    
    if 'purchase_amount' in event_data:
        amount = event_data['purchase_amount']
        if not isinstance(amount, (int, float)) or amount < 0:
            raise ValueError("purchase_amount must be a non-negative number")
    
    if 'session_duration' in event_data:
        duration = event_data['session_duration']
        if not isinstance(duration, (int, float)) or duration < 0:
            raise ValueError("session_duration must be a non-negative number")
    
    return True


def validate_retention_calculation(retention_data: Dict[str, Any]) -> bool:
    """
    Validate retention calculation input data.
    
    Args:
        retention_data: Dictionary containing retention calculation parameters
        
    Returns:
        bool: True if valid, raises ValueError if invalid
        
    Raises:
        ValueError: If validation fails
    """
    required_fields = ['cohort_date', 'retention_period', 'player_count']
    
    # Check required fields
    for field in required_fields:
        if field not in retention_data:
            raise ValueError(f"Missing required field: {field}")
    
    # Validate cohort_date
    cohort_date = retention_data['cohort_date']
    if isinstance(cohort_date, str):
        try:
            date.fromisoformat(cohort_date)
        except ValueError:
            raise ValueError("cohort_date must be a valid ISO format date string")
    elif not isinstance(cohort_date, date):
        raise ValueError("cohort_date must be a date object or ISO format string")
    
    # Validate retention_period
    retention_period = retention_data['retention_period']
    valid_periods = [1, 7, 30]
    if retention_period not in valid_periods:
        raise ValueError(f"retention_period must be one of: {valid_periods}")
    
    # Validate player_count
    player_count = retention_data['player_count']
    if not isinstance(player_count, int) or player_count < 0:
        raise ValueError("player_count must be a non-negative integer")
    
    return True


def validate_metric_thresholds(metrics: Dict[str, float], thresholds: Dict[str, float]) -> List[str]:
    """
    Validate metrics against defined thresholds and return warnings.
    
    Args:
        metrics: Dictionary of metric names to values
        thresholds: Dictionary of metric names to threshold values
        
    Returns:
        List[str]: List of warning messages for metrics below thresholds
    """
    warnings = []
    
    for metric_name, threshold in thresholds.items():
        if metric_name in metrics:
            value = metrics[metric_name]
            if not isinstance(value, (int, float)):
                warnings.append(f"Invalid metric value for {metric_name}: must be numeric")
            elif value < threshold:
                warnings.append(f"Metric {metric_name} ({value:.3f}) is below threshold ({threshold:.3f})")
    
    return warnings


def validate_date_range(start_date: date, end_date: date, max_days: int = 365) -> bool:
    """
    Validate date range for analytics queries.
    
    Args:
        start_date: Start date of the range
        end_date: End date of the range
        max_days: Maximum allowed days in range
        
    Returns:
        bool: True if valid, raises ValueError if invalid
        
    Raises:
        ValueError: If validation fails
    """
    if not isinstance(start_date, date):
        raise ValueError("start_date must be a date object")
    
    if not isinstance(end_date, date):
        raise ValueError("end_date must be a date object")
    
    if start_date > end_date:
        raise ValueError("start_date cannot be after end_date")
    
    days_diff = (end_date - start_date).days
    if days_diff > max_days:
        raise ValueError(f"Date range cannot exceed {max_days} days")
    
    return True


def sanitize_player_segment(segment: str) -> str:
    """
    Sanitize and validate player segment string.
    
    Args:
        segment: Raw segment string
        
    Returns:
        str: Sanitized segment string
        
    Raises:
        ValueError: If segment is invalid
    """
    if not isinstance(segment, str):
        raise ValueError("segment must be a string")
    
    # Remove whitespace and convert to lowercase
    segment = segment.strip().lower()
    
    if not segment:
        raise ValueError("segment cannot be empty")
    
    # Replace spaces with underscores and remove invalid characters
    segment = re.sub(r'[^a-zA-Z0-9_-]', '_', segment)
    segment = re.sub(r'_+', '_', segment)  # Replace multiple underscores with single
    segment = segment.strip('_')  # Remove leading/trailing underscores
    
    if not segment:
        raise ValueError("segment contains no valid characters")
    
    return segment