"""
Unit tests for RetentionMetrics data model.
"""
import pytest
from datetime import date
from src.models.retention_metrics import RetentionMetrics


class TestRetentionMetrics:
    """Test cases for RetentionMetrics model."""
    
    def test_valid_retention_metrics_creation(self):
        """Test creating a valid RetentionMetrics instance."""
        metrics = RetentionMetrics(
            cohort_date=date(2024, 1, 1),
            day_1_retention=0.8,
            day_7_retention=0.6,
            day_30_retention=0.4,
            cohort_size=1000,
            segment="new_users"
        )
        
        assert metrics.cohort_date == date(2024, 1, 1)
        assert metrics.day_1_retention == 0.8
        assert metrics.cohort_size == 1000
        assert metrics.segment == "new_users"
    
    def test_invalid_retention_rate_above_one(self):
        """Test validation fails for retention rate above 1."""
        with pytest.raises(ValueError, match="day_1_retention must be between 0 and 1"):
            RetentionMetrics(
                cohort_date=date(2024, 1, 1),
                day_1_retention=1.2,
                day_7_retention=0.6,
                day_30_retention=0.4,
                cohort_size=1000,
                segment="new_users"
            )
    
    def test_invalid_retention_rate_below_zero(self):
        """Test validation fails for negative retention rate."""
        with pytest.raises(ValueError, match="day_7_retention must be between 0 and 1"):
            RetentionMetrics(
                cohort_date=date(2024, 1, 1),
                day_1_retention=0.8,
                day_7_retention=-0.1,
                day_30_retention=0.4,
                cohort_size=1000,
                segment="new_users"
            )
    
    def test_invalid_retention_logic_day7_exceeds_day1(self):
        """Test validation fails when day 7 retention exceeds day 1."""
        with pytest.raises(ValueError, match="day_7_retention should not exceed day_1_retention"):
            RetentionMetrics(
                cohort_date=date(2024, 1, 1),
                day_1_retention=0.6,
                day_7_retention=0.8,
                day_30_retention=0.4,
                cohort_size=1000,
                segment="new_users"
            )
    
    def test_invalid_retention_logic_day30_exceeds_day7(self):
        """Test validation fails when day 30 retention exceeds day 7."""
        with pytest.raises(ValueError, match="day_30_retention should not exceed day_7_retention"):
            RetentionMetrics(
                cohort_date=date(2024, 1, 1),
                day_1_retention=0.8,
                day_7_retention=0.6,
                day_30_retention=0.7,
                cohort_size=1000,
                segment="new_users"
            )
    
    def test_invalid_cohort_size_zero(self):
        """Test validation fails for zero cohort size."""
        with pytest.raises(ValueError, match="cohort_size must be a positive integer"):
            RetentionMetrics(
                cohort_date=date(2024, 1, 1),
                day_1_retention=0.8,
                day_7_retention=0.6,
                day_30_retention=0.4,
                cohort_size=0,
                segment="new_users"
            )
    
    def test_invalid_segment_empty(self):
        """Test validation fails for empty segment."""
        with pytest.raises(ValueError, match="segment must be a non-empty string"):
            RetentionMetrics(
                cohort_date=date(2024, 1, 1),
                day_1_retention=0.8,
                day_7_retention=0.6,
                day_30_retention=0.4,
                cohort_size=1000,
                segment=""
            )
    
    def test_invalid_segment_format(self):
        """Test validation fails for invalid segment format."""
        with pytest.raises(ValueError, match="segment must contain only alphanumeric characters"):
            RetentionMetrics(
                cohort_date=date(2024, 1, 1),
                day_1_retention=0.8,
                day_7_retention=0.6,
                day_30_retention=0.4,
                cohort_size=1000,
                segment="new users!"
            )
    
    def test_to_dict_serialization(self):
        """Test serialization to dictionary."""
        metrics = RetentionMetrics(
            cohort_date=date(2024, 1, 1),
            day_1_retention=0.8,
            day_7_retention=0.6,
            day_30_retention=0.4,
            cohort_size=1000,
            segment="new_users"
        )
        
        data = metrics.to_dict()
        
        assert data['cohort_date'] == "2024-01-01"
        assert data['day_1_retention'] == 0.8
        assert data['cohort_size'] == 1000
        assert data['segment'] == "new_users"
    
    def test_from_dict_deserialization(self):
        """Test deserialization from dictionary."""
        data = {
            'cohort_date': "2024-01-01",
            'day_1_retention': 0.8,
            'day_7_retention': 0.6,
            'day_30_retention': 0.4,
            'cohort_size': 1000,
            'segment': "new_users"
        }
        
        metrics = RetentionMetrics.from_dict(data)
        
        assert metrics.cohort_date == date(2024, 1, 1)
        assert metrics.day_1_retention == 0.8
        assert metrics.cohort_size == 1000
        assert metrics.segment == "new_users"
    
    def test_get_retention_at_day(self):
        """Test getting retention rate for specific days."""
        metrics = RetentionMetrics(
            cohort_date=date(2024, 1, 1),
            day_1_retention=0.8,
            day_7_retention=0.6,
            day_30_retention=0.4,
            cohort_size=1000,
            segment="new_users"
        )
        
        assert metrics.get_retention_at_day(1) == 0.8
        assert metrics.get_retention_at_day(7) == 0.6
        assert metrics.get_retention_at_day(30) == 0.4
        assert metrics.get_retention_at_day(14) is None
    
    def test_serialization_roundtrip(self):
        """Test that serialization and deserialization preserve data."""
        original = RetentionMetrics(
            cohort_date=date(2024, 1, 1),
            day_1_retention=0.8,
            day_7_retention=0.6,
            day_30_retention=0.4,
            cohort_size=1000,
            segment="new_users"
        )
        
        data = original.to_dict()
        restored = RetentionMetrics.from_dict(data)
        
        assert original.cohort_date == restored.cohort_date
        assert original.day_1_retention == restored.day_1_retention
        assert original.cohort_size == restored.cohort_size
        assert original.segment == restored.segment