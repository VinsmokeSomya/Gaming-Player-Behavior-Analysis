"""
Unit tests for PlayerProfile data model.
"""
import pytest
from datetime import datetime
from src.models.player_profile import PlayerProfile


class TestPlayerProfile:
    """Test cases for PlayerProfile model."""
    
    def test_valid_player_profile_creation(self):
        """Test creating a valid PlayerProfile instance."""
        profile = PlayerProfile(
            player_id="player_123",
            registration_date=datetime(2024, 1, 1, 10, 0, 0),
            last_active_date=datetime(2024, 1, 15, 14, 30, 0),
            total_sessions=25,
            total_playtime_minutes=1500,
            highest_level_reached=10,
            total_purchases=29.99,
            churn_risk_score=0.3,
            churn_prediction_date=datetime(2024, 1, 16, 0, 0, 0)
        )
        
        assert profile.player_id == "player_123"
        assert profile.total_sessions == 25
        assert profile.churn_risk_score == 0.3
    
    def test_invalid_player_id_empty(self):
        """Test validation fails for empty player_id."""
        with pytest.raises(ValueError, match="player_id must be a non-empty string"):
            PlayerProfile(
                player_id="",
                registration_date=datetime(2024, 1, 1),
                last_active_date=datetime(2024, 1, 15),
                total_sessions=25,
                total_playtime_minutes=1500,
                highest_level_reached=10,
                total_purchases=29.99,
                churn_risk_score=0.3,
                churn_prediction_date=datetime(2024, 1, 16)
            )
    
    def test_invalid_player_id_format(self):
        """Test validation fails for invalid player_id format."""
        with pytest.raises(ValueError, match="player_id must contain only alphanumeric characters"):
            PlayerProfile(
                player_id="player@123",
                registration_date=datetime(2024, 1, 1),
                last_active_date=datetime(2024, 1, 15),
                total_sessions=25,
                total_playtime_minutes=1500,
                highest_level_reached=10,
                total_purchases=29.99,
                churn_risk_score=0.3,
                churn_prediction_date=datetime(2024, 1, 16)
            )
    
    def test_invalid_date_order(self):
        """Test validation fails when last_active_date is before registration_date."""
        with pytest.raises(ValueError, match="last_active_date cannot be before registration_date"):
            PlayerProfile(
                player_id="player_123",
                registration_date=datetime(2024, 1, 15),
                last_active_date=datetime(2024, 1, 1),
                total_sessions=25,
                total_playtime_minutes=1500,
                highest_level_reached=10,
                total_purchases=29.99,
                churn_risk_score=0.3,
                churn_prediction_date=datetime(2024, 1, 16)
            )
    
    def test_invalid_negative_sessions(self):
        """Test validation fails for negative total_sessions."""
        with pytest.raises(ValueError, match="total_sessions must be a non-negative integer"):
            PlayerProfile(
                player_id="player_123",
                registration_date=datetime(2024, 1, 1),
                last_active_date=datetime(2024, 1, 15),
                total_sessions=-5,
                total_playtime_minutes=1500,
                highest_level_reached=10,
                total_purchases=29.99,
                churn_risk_score=0.3,
                churn_prediction_date=datetime(2024, 1, 16)
            )
    
    def test_invalid_churn_risk_score_range(self):
        """Test validation fails for churn_risk_score outside 0-1 range."""
        with pytest.raises(ValueError, match="churn_risk_score must be between 0 and 1"):
            PlayerProfile(
                player_id="player_123",
                registration_date=datetime(2024, 1, 1),
                last_active_date=datetime(2024, 1, 15),
                total_sessions=25,
                total_playtime_minutes=1500,
                highest_level_reached=10,
                total_purchases=29.99,
                churn_risk_score=1.5,
                churn_prediction_date=datetime(2024, 1, 16)
            )
    
    def test_to_dict_serialization(self):
        """Test serialization to dictionary."""
        profile = PlayerProfile(
            player_id="player_123",
            registration_date=datetime(2024, 1, 1, 10, 0, 0),
            last_active_date=datetime(2024, 1, 15, 14, 30, 0),
            total_sessions=25,
            total_playtime_minutes=1500,
            highest_level_reached=10,
            total_purchases=29.99,
            churn_risk_score=0.3,
            churn_prediction_date=datetime(2024, 1, 16, 0, 0, 0)
        )
        
        data = profile.to_dict()
        
        assert data['player_id'] == "player_123"
        assert data['total_sessions'] == 25
        assert data['churn_risk_score'] == 0.3
        assert data['registration_date'] == "2024-01-01T10:00:00"
    
    def test_from_dict_deserialization(self):
        """Test deserialization from dictionary."""
        data = {
            'player_id': "player_123",
            'registration_date': "2024-01-01T10:00:00",
            'last_active_date': "2024-01-15T14:30:00",
            'total_sessions': 25,
            'total_playtime_minutes': 1500,
            'highest_level_reached': 10,
            'total_purchases': 29.99,
            'churn_risk_score': 0.3,
            'churn_prediction_date': "2024-01-16T00:00:00"
        }
        
        profile = PlayerProfile.from_dict(data)
        
        assert profile.player_id == "player_123"
        assert profile.total_sessions == 25
        assert profile.churn_risk_score == 0.3
        assert profile.registration_date == datetime(2024, 1, 1, 10, 0, 0)
    
    def test_serialization_roundtrip(self):
        """Test that serialization and deserialization preserve data."""
        original = PlayerProfile(
            player_id="player_123",
            registration_date=datetime(2024, 1, 1, 10, 0, 0),
            last_active_date=datetime(2024, 1, 15, 14, 30, 0),
            total_sessions=25,
            total_playtime_minutes=1500,
            highest_level_reached=10,
            total_purchases=29.99,
            churn_risk_score=0.3,
            churn_prediction_date=datetime(2024, 1, 16, 0, 0, 0)
        )
        
        data = original.to_dict()
        restored = PlayerProfile.from_dict(data)
        
        assert original.player_id == restored.player_id
        assert original.registration_date == restored.registration_date
        assert original.total_sessions == restored.total_sessions
        assert original.churn_risk_score == restored.churn_risk_score