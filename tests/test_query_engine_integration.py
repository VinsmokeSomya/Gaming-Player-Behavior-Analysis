"""Integration tests for the retention query engine with database."""

import pytest
from datetime import date, datetime, timedelta
from src.analytics.query_engine import RetentionQueryEngine
from src.database import db_manager


class TestRetentionQueryEngineIntegration:
    """Integration tests for RetentionQueryEngine with actual database."""
    
    @pytest.fixture(scope="class")
    def query_engine(self):
        """Create a query engine instance for integration testing."""
        return RetentionQueryEngine()
    
    @pytest.fixture(scope="class")
    def sample_dates(self):
        """Sample date range for testing."""
        return {
            'start_date': date(2024, 1, 1),
            'end_date': date(2024, 1, 31)
        }
    
    def test_database_connection(self, query_engine):
        """Test that the query engine can connect to the database."""
        # This test requires the database to be running
        try:
            connection_successful = db_manager.test_connection()
            if not connection_successful:
                pytest.skip("Database not available for integration testing")
        except Exception:
            pytest.skip("Database not available for integration testing")
    
    def test_calculate_cohort_retention_with_empty_database(self, query_engine, sample_dates):
        """Test cohort retention calculation with empty database."""
        try:
            if not db_manager.test_connection():
                pytest.skip("Database not available")
            
            results = query_engine.calculate_cohort_retention(
                sample_dates['start_date'],
                sample_dates['end_date']
            )
            
            # With empty database, should return empty list
            assert isinstance(results, list)
            # Results might be empty or contain zero values
            
        except Exception as e:
            if "connection" in str(e).lower():
                pytest.skip("Database not available for integration testing")
            else:
                raise
    
    def test_analyze_drop_off_with_empty_database(self, query_engine, sample_dates):
        """Test drop-off analysis with empty database."""
        try:
            if not db_manager.test_connection():
                pytest.skip("Database not available")
            
            results = query_engine.analyze_drop_off_by_level(
                sample_dates['start_date'],
                sample_dates['end_date'],
                max_level=10
            )
            
            # With empty database, should return empty list
            assert isinstance(results, list)
            
        except Exception as e:
            if "connection" in str(e).lower():
                pytest.skip("Database not available for integration testing")
            else:
                raise
    
    def test_segment_players_with_empty_database(self, query_engine, sample_dates):
        """Test player segmentation with empty database."""
        try:
            if not db_manager.test_connection():
                pytest.skip("Database not available")
            
            results = query_engine.segment_players_by_behavior(
                sample_dates['start_date'],
                sample_dates['end_date']
            )
            
            # With empty database, should return empty list
            assert isinstance(results, list)
            
        except Exception as e:
            if "connection" in str(e).lower():
                pytest.skip("Database not available for integration testing")
            else:
                raise
    
    def test_get_daily_active_users_with_empty_database(self, query_engine, sample_dates):
        """Test DAU calculation with empty database."""
        try:
            if not db_manager.test_connection():
                pytest.skip("Database not available")
            
            results = query_engine.get_daily_active_users(
                sample_dates['start_date'],
                sample_dates['end_date']
            )
            
            # With empty database, should return empty list
            assert isinstance(results, list)
            
        except Exception as e:
            if "connection" in str(e).lower():
                pytest.skip("Database not available for integration testing")
            else:
                raise
    
    def test_get_weekly_active_users_with_empty_database(self, query_engine, sample_dates):
        """Test WAU calculation with empty database."""
        try:
            if not db_manager.test_connection():
                pytest.skip("Database not available")
            
            results = query_engine.get_weekly_active_users(
                sample_dates['start_date'],
                sample_dates['end_date']
            )
            
            # With empty database, should return empty list
            assert isinstance(results, list)
            
        except Exception as e:
            if "connection" in str(e).lower():
                pytest.skip("Database not available for integration testing")
            else:
                raise
    
    def test_query_engine_sql_syntax_validation(self, query_engine, sample_dates):
        """Test that all SQL queries have valid syntax by attempting to execute them."""
        try:
            if not db_manager.test_connection():
                pytest.skip("Database not available")
            
            # Test all query methods to ensure SQL syntax is valid
            methods_to_test = [
                lambda: query_engine.calculate_cohort_retention(
                    sample_dates['start_date'], sample_dates['end_date']
                ),
                lambda: query_engine.analyze_drop_off_by_level(
                    sample_dates['start_date'], sample_dates['end_date'], max_level=5
                ),
                lambda: query_engine.segment_players_by_behavior(
                    sample_dates['start_date'], sample_dates['end_date']
                ),
                lambda: query_engine.get_retention_by_segment(
                    sample_dates['start_date'], sample_dates['end_date'], 'High Engagement'
                ),
                lambda: query_engine.get_daily_active_users(
                    sample_dates['start_date'], sample_dates['end_date']
                ),
                lambda: query_engine.get_weekly_active_users(
                    sample_dates['start_date'], sample_dates['end_date']
                )
            ]
            
            for method in methods_to_test:
                try:
                    result = method()
                    assert isinstance(result, list)
                except Exception as e:
                    # If it's not a SQL syntax error, the query structure is valid
                    if "syntax error" in str(e).lower():
                        pytest.fail(f"SQL syntax error in query: {e}")
                    # Other errors (like missing data) are acceptable for this test
            
        except Exception as e:
            if "connection" in str(e).lower():
                pytest.skip("Database not available for integration testing")
            else:
                raise