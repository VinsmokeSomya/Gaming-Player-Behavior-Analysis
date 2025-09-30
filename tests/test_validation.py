"""
Unit tests for validation functions.
"""
import pytest
from datetime import datetime, date
from src.models.validation import (
    validate_player_event,
    validate_retention_calculation,
    validate_metric_thresholds,
    validate_date_range,
    sanitize_player_segment
)


class TestValidatePlayerEvent:
    """Test cases for validate_player_event function."""
    
    def test_valid_player_event(self):
        """Test validation passes for valid player event."""
        event_data = {
            'player_id': 'player_123',
            'event_type': 'session_start',
            'timestamp': datetime(2024, 1, 15, 10, 30, 0)
        }
        
        assert validate_player_event(event_data) is True
    
    def test_valid_player_event_with_string_timestamp(self):
        """Test validation passes for valid event with string timestamp."""
        event_data = {
            'player_id': 'player_123',
            'event_type': 'level_complete',
            'timestamp': '2024-01-15T10:30:00',
            'level': 5
        }
        
        assert validate_player_event(event_data) is True
    
    def test_missing_required_field(self):
        """Test validation fails when required field is missing."""
        event_data = {
            'player_id': 'player_123',
            'event_type': 'session_start'
            # Missing timestamp
        }
        
        with pytest.raises(ValueError, match="Missing required field: timestamp"):
            validate_player_event(event_data)
    
    def test_invalid_player_id_format(self):
        """Test validation fails for invalid player_id format."""
        event_data = {
            'player_id': 'player@123',
            'event_type': 'session_start',
            'timestamp': datetime(2024, 1, 15, 10, 30, 0)
        }
        
        with pytest.raises(ValueError, match="player_id must contain only alphanumeric characters"):
            validate_player_event(event_data)
    
    def test_invalid_event_type(self):
        """Test validation fails for invalid event_type."""
        event_data = {
            'player_id': 'player_123',
            'event_type': 'invalid_event',
            'timestamp': datetime(2024, 1, 15, 10, 30, 0)
        }
        
        with pytest.raises(ValueError, match="event_type must be one of"):
            validate_player_event(event_data)
    
    def test_invalid_timestamp_format(self):
        """Test validation fails for invalid timestamp format."""
        event_data = {
            'player_id': 'player_123',
            'event_type': 'session_start',
            'timestamp': 'invalid-timestamp'
        }
        
        with pytest.raises(ValueError, match="timestamp must be a valid ISO format datetime string"):
            validate_player_event(event_data)
    
    def test_invalid_level_negative(self):
        """Test validation fails for negative level."""
        event_data = {
            'player_id': 'player_123',
            'event_type': 'level_complete',
            'timestamp': datetime(2024, 1, 15, 10, 30, 0),
            'level': -1
        }
        
        with pytest.raises(ValueError, match="level must be a positive integer"):
            validate_player_event(event_data)
    
    def test_invalid_purchase_amount_negative(self):
        """Test validation fails for negative purchase amount."""
        event_data = {
            'player_id': 'player_123',
            'event_type': 'purchase',
            'timestamp': datetime(2024, 1, 15, 10, 30, 0),
            'purchase_amount': -5.99
        }
        
        with pytest.raises(ValueError, match="purchase_amount must be a non-negative number"):
            validate_player_event(event_data)


class TestValidateRetentionCalculation:
    """Test cases for validate_retention_calculation function."""
    
    def test_valid_retention_calculation(self):
        """Test validation passes for valid retention calculation data."""
        retention_data = {
            'cohort_date': date(2024, 1, 1),
            'retention_period': 7,
            'player_count': 1000
        }
        
        assert validate_retention_calculation(retention_data) is True
    
    def test_valid_retention_calculation_string_date(self):
        """Test validation passes with string date."""
        retention_data = {
            'cohort_date': '2024-01-01',
            'retention_period': 30,
            'player_count': 500
        }
        
        assert validate_retention_calculation(retention_data) is True
    
    def test_missing_required_field(self):
        """Test validation fails when required field is missing."""
        retention_data = {
            'cohort_date': date(2024, 1, 1),
            'retention_period': 7
            # Missing player_count
        }
        
        with pytest.raises(ValueError, match="Missing required field: player_count"):
            validate_retention_calculation(retention_data)
    
    def test_invalid_retention_period(self):
        """Test validation fails for invalid retention period."""
        retention_data = {
            'cohort_date': date(2024, 1, 1),
            'retention_period': 14,  # Not in valid periods
            'player_count': 1000
        }
        
        with pytest.raises(ValueError, match="retention_period must be one of"):
            validate_retention_calculation(retention_data)
    
    def test_invalid_player_count_negative(self):
        """Test validation fails for negative player count."""
        retention_data = {
            'cohort_date': date(2024, 1, 1),
            'retention_period': 7,
            'player_count': -100
        }
        
        with pytest.raises(ValueError, match="player_count must be a non-negative integer"):
            validate_retention_calculation(retention_data)


class TestValidateMetricThresholds:
    """Test cases for validate_metric_thresholds function."""
    
    def test_metrics_above_thresholds(self):
        """Test no warnings when metrics are above thresholds."""
        metrics = {
            'day_1_retention': 0.8,
            'day_7_retention': 0.6,
            'churn_rate': 0.2
        }
        thresholds = {
            'day_1_retention': 0.7,
            'day_7_retention': 0.5,
            'churn_rate': 0.1  # Lower threshold so 0.2 is above it
        }
        
        warnings = validate_metric_thresholds(metrics, thresholds)
        assert len(warnings) == 0
    
    def test_metrics_below_thresholds(self):
        """Test warnings generated when metrics are below thresholds."""
        metrics = {
            'day_1_retention': 0.6,
            'day_7_retention': 0.4,
            'churn_rate': 0.2  # Below threshold of 0.3
        }
        thresholds = {
            'day_1_retention': 0.7,
            'day_7_retention': 0.5,
            'churn_rate': 0.3
        }
        
        warnings = validate_metric_thresholds(metrics, thresholds)
        assert len(warnings) == 3
        assert "day_1_retention" in warnings[0]
        assert "below threshold" in warnings[0]
    
    def test_invalid_metric_value(self):
        """Test warning for invalid metric value."""
        metrics = {
            'day_1_retention': 'invalid',
            'day_7_retention': 0.6
        }
        thresholds = {
            'day_1_retention': 0.7,
            'day_7_retention': 0.5
        }
        
        warnings = validate_metric_thresholds(metrics, thresholds)
        assert len(warnings) == 1
        assert "Invalid metric value" in warnings[0]


class TestValidateDateRange:
    """Test cases for validate_date_range function."""
    
    def test_valid_date_range(self):
        """Test validation passes for valid date range."""
        start_date = date(2024, 1, 1)
        end_date = date(2024, 1, 31)
        
        assert validate_date_range(start_date, end_date) is True
    
    def test_invalid_date_order(self):
        """Test validation fails when start_date is after end_date."""
        start_date = date(2024, 1, 31)
        end_date = date(2024, 1, 1)
        
        with pytest.raises(ValueError, match="start_date cannot be after end_date"):
            validate_date_range(start_date, end_date)
    
    def test_date_range_too_large(self):
        """Test validation fails when date range exceeds maximum days."""
        start_date = date(2024, 1, 1)
        end_date = date(2025, 1, 1)  # 366 days
        
        with pytest.raises(ValueError, match="Date range cannot exceed 365 days"):
            validate_date_range(start_date, end_date, max_days=365)
    
    def test_same_date_range(self):
        """Test validation passes for same start and end date."""
        test_date = date(2024, 1, 1)
        
        assert validate_date_range(test_date, test_date) is True


class TestSanitizePlayerSegment:
    """Test cases for sanitize_player_segment function."""
    
    def test_valid_segment(self):
        """Test sanitization of valid segment."""
        result = sanitize_player_segment("new_users")
        assert result == "new_users"
    
    def test_segment_with_spaces(self):
        """Test sanitization replaces spaces with underscores."""
        result = sanitize_player_segment("new users")
        assert result == "new_users"
    
    def test_segment_with_special_characters(self):
        """Test sanitization removes special characters."""
        result = sanitize_player_segment("new-users@2024!")
        assert result == "new-users_2024"
    
    def test_segment_case_conversion(self):
        """Test sanitization converts to lowercase."""
        result = sanitize_player_segment("NEW_USERS")
        assert result == "new_users"
    
    def test_segment_multiple_underscores(self):
        """Test sanitization consolidates multiple underscores."""
        result = sanitize_player_segment("new___users")
        assert result == "new_users"
    
    def test_empty_segment(self):
        """Test validation fails for empty segment."""
        with pytest.raises(ValueError, match="segment cannot be empty"):
            sanitize_player_segment("")
    
    def test_segment_only_special_characters(self):
        """Test validation fails for segment with only special characters."""
        with pytest.raises(ValueError, match="segment contains no valid characters"):
            sanitize_player_segment("!@#$%")