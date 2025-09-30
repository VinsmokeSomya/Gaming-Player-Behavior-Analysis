"""Unit tests for the retention query engine."""

import pytest
from datetime import date, datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from sqlalchemy.orm import Session

from src.analytics.query_engine import (
    RetentionQueryEngine,
    RetentionQueryResult,
    DropOffAnalysisResult,
    PlayerSegmentResult
)


class TestRetentionQueryEngine:
    """Test cases for RetentionQueryEngine."""
    
    @pytest.fixture
    def query_engine(self):
        """Create a query engine instance for testing."""
        return RetentionQueryEngine()
    
    @pytest.fixture
    def mock_session(self):
        """Create a mock database session."""
        session = Mock(spec=Session)
        return session
    
    @pytest.fixture
    def sample_dates(self):
        """Sample date range for testing."""
        return {
            'start_date': date(2024, 1, 1),
            'end_date': date(2024, 1, 31)
        }
    
    def test_calculate_cohort_retention_success(self, query_engine, mock_session, sample_dates):
        """Test successful cohort retention calculation."""
        # Mock database response
        mock_result = Mock()
        mock_result.fetchall.return_value = [
            Mock(
                cohort_date=date(2024, 1, 1),
                day_1_retention=0.8,
                day_7_retention=0.6,
                day_30_retention=0.4,
                cohort_size=100
            ),
            Mock(
                cohort_date=date(2024, 1, 2),
                day_1_retention=0.75,
                day_7_retention=0.55,
                day_30_retention=0.35,
                cohort_size=120
            )
        ]
        
        mock_session.execute.return_value = mock_result
        
        with patch.object(query_engine.db_manager, 'get_session') as mock_get_session:
            mock_get_session.return_value.__enter__.return_value = mock_session
            
            results = query_engine.calculate_cohort_retention(
                sample_dates['start_date'],
                sample_dates['end_date']
            )
        
        assert len(results) == 2
        assert isinstance(results[0], RetentionQueryResult)
        assert results[0].cohort_date == date(2024, 1, 1)
        assert results[0].day_1_retention == 0.8
        assert results[0].day_7_retention == 0.6
        assert results[0].day_30_retention == 0.4
        assert results[0].cohort_size == 100
        
        assert results[1].cohort_date == date(2024, 1, 2)
        assert results[1].cohort_size == 120
    
    def test_calculate_cohort_retention_with_segment(self, query_engine, mock_session, sample_dates):
        """Test cohort retention calculation with segment filter."""
        mock_result = Mock()
        mock_result.fetchall.return_value = [
            Mock(
                cohort_date=date(2024, 1, 1),
                day_1_retention=0.9,
                day_7_retention=0.7,
                day_30_retention=0.5,
                cohort_size=50
            )
        ]
        
        mock_session.execute.return_value = mock_result
        
        with patch.object(query_engine.db_manager, 'get_session') as mock_get_session:
            mock_get_session.return_value.__enter__.return_value = mock_session
            
            results = query_engine.calculate_cohort_retention(
                sample_dates['start_date'],
                sample_dates['end_date'],
                segment='High Engagement'
            )
        
        assert len(results) == 1
        assert results[0].segment == 'High Engagement'
        assert results[0].day_1_retention == 0.9
    
    def test_analyze_drop_off_by_level_success(self, query_engine, mock_session, sample_dates):
        """Test successful drop-off analysis by level."""
        mock_result = Mock()
        mock_result.fetchall.return_value = [
            Mock(
                level=1,
                players_reached=1000,
                players_completed=800,
                drop_off_rate=0.2,
                completion_rate=0.8
            ),
            Mock(
                level=2,
                players_reached=800,
                players_completed=600,
                drop_off_rate=0.25,
                completion_rate=0.75
            ),
            Mock(
                level=3,
                players_reached=600,
                players_completed=400,
                drop_off_rate=0.33,
                completion_rate=0.67
            )
        ]
        
        mock_session.execute.return_value = mock_result
        
        with patch.object(query_engine.db_manager, 'get_session') as mock_get_session:
            mock_get_session.return_value.__enter__.return_value = mock_session
            
            results = query_engine.analyze_drop_off_by_level(
                sample_dates['start_date'],
                sample_dates['end_date'],
                max_level=10
            )
        
        assert len(results) == 3
        assert isinstance(results[0], DropOffAnalysisResult)
        
        # Verify level 1 results
        assert results[0].level == 1
        assert results[0].players_reached == 1000
        assert results[0].players_completed == 800
        assert results[0].drop_off_rate == 0.2
        assert results[0].completion_rate == 0.8
        
        # Verify level 3 has higher drop-off rate
        assert results[2].drop_off_rate > results[0].drop_off_rate
    
    def test_segment_players_by_behavior_success(self, query_engine, mock_session, sample_dates):
        """Test successful player segmentation by behavior."""
        mock_result = Mock()
        mock_result.fetchall.return_value = [
            Mock(
                segment='High Engagement',
                player_count=250,
                avg_sessions=25.5,
                avg_playtime=1500.0,
                avg_retention_day_7=0.85
            ),
            Mock(
                segment='Medium Engagement',
                player_count=400,
                avg_sessions=15.2,
                avg_playtime=800.0,
                avg_retention_day_7=0.65
            ),
            Mock(
                segment='Low Engagement',
                player_count=300,
                avg_sessions=5.8,
                avg_playtime=300.0,
                avg_retention_day_7=0.35
            ),
            Mock(
                segment='Minimal Engagement',
                player_count=150,
                avg_sessions=2.1,
                avg_playtime=120.0,
                avg_retention_day_7=0.15
            )
        ]
        
        mock_session.execute.return_value = mock_result
        
        with patch.object(query_engine.db_manager, 'get_session') as mock_get_session:
            mock_get_session.return_value.__enter__.return_value = mock_session
            
            results = query_engine.segment_players_by_behavior(
                sample_dates['start_date'],
                sample_dates['end_date']
            )
        
        assert len(results) == 4
        assert isinstance(results[0], PlayerSegmentResult)
        
        # Verify segments are ordered correctly
        segments = [r.segment for r in results]
        expected_segments = ['High Engagement', 'Medium Engagement', 'Low Engagement', 'Minimal Engagement']
        assert segments == expected_segments
        
        # Verify high engagement segment has best metrics
        high_engagement = results[0]
        assert high_engagement.avg_retention_day_7 == 0.85
        assert high_engagement.avg_sessions == 25.5
        
        # Verify minimal engagement has worst retention
        minimal_engagement = results[3]
        assert minimal_engagement.avg_retention_day_7 == 0.15
    
    def test_get_retention_by_segment_success(self, query_engine, mock_session, sample_dates):
        """Test getting retention rates filtered by segment."""
        mock_result = Mock()
        mock_result.fetchall.return_value = [
            Mock(
                cohort_date=date(2024, 1, 1),
                day_1_retention=0.95,
                day_7_retention=0.85,
                day_30_retention=0.65,
                cohort_size=50
            )
        ]
        
        mock_session.execute.return_value = mock_result
        
        with patch.object(query_engine.db_manager, 'get_session') as mock_get_session:
            mock_get_session.return_value.__enter__.return_value = mock_session
            
            results = query_engine.get_retention_by_segment(
                sample_dates['start_date'],
                sample_dates['end_date'],
                'High Engagement'
            )
        
        assert len(results) == 1
        assert results[0].segment == 'High Engagement'
        assert results[0].day_1_retention == 0.95
        assert results[0].day_7_retention == 0.85
        assert results[0].day_30_retention == 0.65
    
    def test_get_daily_active_users_success(self, query_engine, mock_session, sample_dates):
        """Test getting daily active user counts."""
        mock_result = Mock()
        mock_result.fetchall.return_value = [
            Mock(activity_date=date(2024, 1, 1), dau_count=1500),
            Mock(activity_date=date(2024, 1, 2), dau_count=1600),
            Mock(activity_date=date(2024, 1, 3), dau_count=1400)
        ]
        
        mock_session.execute.return_value = mock_result
        
        with patch.object(query_engine.db_manager, 'get_session') as mock_get_session:
            mock_get_session.return_value.__enter__.return_value = mock_session
            
            results = query_engine.get_daily_active_users(
                sample_dates['start_date'],
                sample_dates['end_date']
            )
        
        assert len(results) == 3
        assert results[0] == (date(2024, 1, 1), 1500)
        assert results[1] == (date(2024, 1, 2), 1600)
        assert results[2] == (date(2024, 1, 3), 1400)
    
    def test_get_weekly_active_users_success(self, query_engine, mock_session, sample_dates):
        """Test getting weekly active user counts."""
        mock_result = Mock()
        mock_result.fetchall.return_value = [
            Mock(week_start=date(2024, 1, 1), wau_count=5000),
            Mock(week_start=date(2024, 1, 8), wau_count=5200),
            Mock(week_start=date(2024, 1, 15), wau_count=4800)
        ]
        
        mock_session.execute.return_value = mock_result
        
        with patch.object(query_engine.db_manager, 'get_session') as mock_get_session:
            mock_get_session.return_value.__enter__.return_value = mock_session
            
            results = query_engine.get_weekly_active_users(
                sample_dates['start_date'],
                sample_dates['end_date']
            )
        
        assert len(results) == 3
        assert results[0] == (date(2024, 1, 1), 5000)
        assert results[1] == (date(2024, 1, 8), 5200)
        assert results[2] == (date(2024, 1, 15), 4800)
    
    def test_calculate_cohort_retention_database_error(self, query_engine, mock_session, sample_dates):
        """Test handling of database errors in cohort retention calculation."""
        mock_session.execute.side_effect = Exception("Database connection failed")
        
        with patch.object(query_engine.db_manager, 'get_session') as mock_get_session:
            mock_get_session.return_value.__enter__.return_value = mock_session
            
            with pytest.raises(Exception) as exc_info:
                query_engine.calculate_cohort_retention(
                    sample_dates['start_date'],
                    sample_dates['end_date']
                )
            
            assert "Database connection failed" in str(exc_info.value)
    
    def test_analyze_drop_off_database_error(self, query_engine, mock_session, sample_dates):
        """Test handling of database errors in drop-off analysis."""
        mock_session.execute.side_effect = Exception("Query timeout")
        
        with patch.object(query_engine.db_manager, 'get_session') as mock_get_session:
            mock_get_session.return_value.__enter__.return_value = mock_session
            
            with pytest.raises(Exception) as exc_info:
                query_engine.analyze_drop_off_by_level(
                    sample_dates['start_date'],
                    sample_dates['end_date']
                )
            
            assert "Query timeout" in str(exc_info.value)
    
    def test_segment_players_database_error(self, query_engine, mock_session, sample_dates):
        """Test handling of database errors in player segmentation."""
        mock_session.execute.side_effect = Exception("Invalid query")
        
        with patch.object(query_engine.db_manager, 'get_session') as mock_get_session:
            mock_get_session.return_value.__enter__.return_value = mock_session
            
            with pytest.raises(Exception) as exc_info:
                query_engine.segment_players_by_behavior(
                    sample_dates['start_date'],
                    sample_dates['end_date']
                )
            
            assert "Invalid query" in str(exc_info.value)
    
    def test_empty_result_handling(self, query_engine, mock_session, sample_dates):
        """Test handling of empty query results."""
        mock_result = Mock()
        mock_result.fetchall.return_value = []
        mock_session.execute.return_value = mock_result
        
        with patch.object(query_engine.db_manager, 'get_session') as mock_get_session:
            mock_get_session.return_value.__enter__.return_value = mock_session
            
            results = query_engine.calculate_cohort_retention(
                sample_dates['start_date'],
                sample_dates['end_date']
            )
        
        assert results == []
    
    def test_query_parameters_passed_correctly(self, query_engine, mock_session, sample_dates):
        """Test that query parameters are passed correctly to the database."""
        mock_result = Mock()
        mock_result.fetchall.return_value = []
        mock_session.execute.return_value = mock_result
        
        with patch.object(query_engine.db_manager, 'get_session') as mock_get_session:
            mock_get_session.return_value.__enter__.return_value = mock_session
            
            query_engine.calculate_cohort_retention(
                sample_dates['start_date'],
                sample_dates['end_date']
            )
            
            # Verify execute was called with correct parameters
            call_args = mock_session.execute.call_args
            assert call_args is not None
            
            # Check that parameters were passed
            params = call_args[0][1]  # Second argument should be parameters
            assert params['start_date'] == sample_dates['start_date']
            assert params['end_date'] == sample_dates['end_date']
    
    def test_drop_off_analysis_max_level_parameter(self, query_engine, mock_session, sample_dates):
        """Test that max_level parameter is correctly passed to drop-off analysis."""
        mock_result = Mock()
        mock_result.fetchall.return_value = []
        mock_session.execute.return_value = mock_result
        
        with patch.object(query_engine.db_manager, 'get_session') as mock_get_session:
            mock_get_session.return_value.__enter__.return_value = mock_session
            
            query_engine.analyze_drop_off_by_level(
                sample_dates['start_date'],
                sample_dates['end_date'],
                max_level=25
            )
            
            # Verify max_level parameter was passed
            call_args = mock_session.execute.call_args
            params = call_args[0][1]
            assert params['max_level'] == 25


class TestRetentionQueryResults:
    """Test cases for result data classes."""
    
    def test_retention_query_result_creation(self):
        """Test RetentionQueryResult creation and attributes."""
        result = RetentionQueryResult(
            cohort_date=date(2024, 1, 1),
            day_1_retention=0.8,
            day_7_retention=0.6,
            day_30_retention=0.4,
            cohort_size=100,
            segment='High Engagement'
        )
        
        assert result.cohort_date == date(2024, 1, 1)
        assert result.day_1_retention == 0.8
        assert result.day_7_retention == 0.6
        assert result.day_30_retention == 0.4
        assert result.cohort_size == 100
        assert result.segment == 'High Engagement'
    
    def test_drop_off_analysis_result_creation(self):
        """Test DropOffAnalysisResult creation and attributes."""
        result = DropOffAnalysisResult(
            level=5,
            players_reached=500,
            players_completed=400,
            drop_off_rate=0.2,
            completion_rate=0.8
        )
        
        assert result.level == 5
        assert result.players_reached == 500
        assert result.players_completed == 400
        assert result.drop_off_rate == 0.2
        assert result.completion_rate == 0.8
    
    def test_player_segment_result_creation(self):
        """Test PlayerSegmentResult creation and attributes."""
        result = PlayerSegmentResult(
            segment='Medium Engagement',
            player_count=300,
            avg_sessions=12.5,
            avg_playtime=750.0,
            avg_retention_day_7=0.55
        )
        
        assert result.segment == 'Medium Engagement'
        assert result.player_count == 300
        assert result.avg_sessions == 12.5
        assert result.avg_playtime == 750.0
        assert result.avg_retention_day_7 == 0.55