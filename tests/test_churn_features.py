"""
Unit tests for ChurnFeatures data model.
"""
import pytest
from datetime import date
from src.models.churn_features import ChurnFeatures


class TestChurnFeatures:
    """Test cases for ChurnFeatures model."""
    
    def test_valid_churn_features_creation(self):
        """Test creating a valid ChurnFeatures instance."""
        features = ChurnFeatures(
            player_id="player_123",
            days_since_last_session=2,
            sessions_last_7_days=5,
            avg_session_duration_minutes=15.5,
            levels_completed_last_week=3,
            purchases_last_30_days=9.99,
            social_connections=12,
            feature_date=date(2024, 1, 15)
        )
        
        assert features.player_id == "player_123"
        assert features.days_since_last_session == 2
        assert features.sessions_last_7_days == 5
        assert features.avg_session_duration_minutes == 15.5
    
    def test_invalid_player_id_empty(self):
        """Test validation fails for empty player_id."""
        with pytest.raises(ValueError, match="player_id must be a non-empty string"):
            ChurnFeatures(
                player_id="",
                days_since_last_session=2,
                sessions_last_7_days=5,
                avg_session_duration_minutes=15.5,
                levels_completed_last_week=3,
                purchases_last_30_days=9.99,
                social_connections=12,
                feature_date=date(2024, 1, 15)
            )
    
    def test_invalid_negative_days_since_last_session(self):
        """Test validation fails for negative days_since_last_session."""
        with pytest.raises(ValueError, match="days_since_last_session must be a non-negative integer"):
            ChurnFeatures(
                player_id="player_123",
                days_since_last_session=-1,
                sessions_last_7_days=5,
                avg_session_duration_minutes=15.5,
                levels_completed_last_week=3,
                purchases_last_30_days=9.99,
                social_connections=12,
                feature_date=date(2024, 1, 15)
            )
    
    def test_invalid_negative_sessions(self):
        """Test validation fails for negative sessions_last_7_days."""
        with pytest.raises(ValueError, match="sessions_last_7_days must be a non-negative integer"):
            ChurnFeatures(
                player_id="player_123",
                days_since_last_session=2,
                sessions_last_7_days=-1,
                avg_session_duration_minutes=15.5,
                levels_completed_last_week=3,
                purchases_last_30_days=9.99,
                social_connections=12,
                feature_date=date(2024, 1, 15)
            )
    
    def test_invalid_business_logic_sessions_vs_days(self):
        """Test validation fails when sessions exist but last session was too long ago."""
        with pytest.raises(ValueError, match="Cannot have sessions in last 7 days if last session was more than 7 days ago"):
            ChurnFeatures(
                player_id="player_123",
                days_since_last_session=10,
                sessions_last_7_days=2,
                avg_session_duration_minutes=15.5,
                levels_completed_last_week=3,
                purchases_last_30_days=9.99,
                social_connections=12,
                feature_date=date(2024, 1, 15)
            )
    
    def test_invalid_negative_purchases(self):
        """Test validation fails for negative purchases."""
        with pytest.raises(ValueError, match="purchases_last_30_days must be a non-negative number"):
            ChurnFeatures(
                player_id="player_123",
                days_since_last_session=2,
                sessions_last_7_days=5,
                avg_session_duration_minutes=15.5,
                levels_completed_last_week=3,
                purchases_last_30_days=-5.0,
                social_connections=12,
                feature_date=date(2024, 1, 15)
            )
    
    def test_to_dict_serialization(self):
        """Test serialization to dictionary."""
        features = ChurnFeatures(
            player_id="player_123",
            days_since_last_session=2,
            sessions_last_7_days=5,
            avg_session_duration_minutes=15.5,
            levels_completed_last_week=3,
            purchases_last_30_days=9.99,
            social_connections=12,
            feature_date=date(2024, 1, 15)
        )
        
        data = features.to_dict()
        
        assert data['player_id'] == "player_123"
        assert data['days_since_last_session'] == 2
        assert data['sessions_last_7_days'] == 5
        assert data['feature_date'] == "2024-01-15"
    
    def test_from_dict_deserialization(self):
        """Test deserialization from dictionary."""
        data = {
            'player_id': "player_123",
            'days_since_last_session': 2,
            'sessions_last_7_days': 5,
            'avg_session_duration_minutes': 15.5,
            'levels_completed_last_week': 3,
            'purchases_last_30_days': 9.99,
            'social_connections': 12,
            'feature_date': "2024-01-15"
        }
        
        features = ChurnFeatures.from_dict(data)
        
        assert features.player_id == "player_123"
        assert features.days_since_last_session == 2
        assert features.sessions_last_7_days == 5
        assert features.feature_date == date(2024, 1, 15)
    
    def test_is_high_churn_risk_true(self):
        """Test high churn risk detection returns True for at-risk player."""
        features = ChurnFeatures(
            player_id="player_123",
            days_since_last_session=5,  # Risk factor
            sessions_last_7_days=0,     # Risk factor (2 points)
            avg_session_duration_minutes=3.0,  # Risk factor
            levels_completed_last_week=0,      # Risk factor
            purchases_last_30_days=0.0,        # Risk factor
            social_connections=2,
            feature_date=date(2024, 1, 15)
        )
        
        assert features.is_high_churn_risk() is True
    
    def test_is_high_churn_risk_false(self):
        """Test high churn risk detection returns False for engaged player."""
        features = ChurnFeatures(
            player_id="player_123",
            days_since_last_session=1,  # No risk
            sessions_last_7_days=5,     # No risk
            avg_session_duration_minutes=15.0,  # No risk
            levels_completed_last_week=3,       # No risk
            purchases_last_30_days=9.99,        # No risk
            social_connections=12,
            feature_date=date(2024, 1, 15)
        )
        
        assert features.is_high_churn_risk() is False
    
    def test_serialization_roundtrip(self):
        """Test that serialization and deserialization preserve data."""
        original = ChurnFeatures(
            player_id="player_123",
            days_since_last_session=2,
            sessions_last_7_days=5,
            avg_session_duration_minutes=15.5,
            levels_completed_last_week=3,
            purchases_last_30_days=9.99,
            social_connections=12,
            feature_date=date(2024, 1, 15)
        )
        
        data = original.to_dict()
        restored = ChurnFeatures.from_dict(data)
        
        assert original.player_id == restored.player_id
        assert original.days_since_last_session == restored.days_since_last_session
        assert original.sessions_last_7_days == restored.sessions_last_7_days
        assert original.feature_date == restored.feature_date