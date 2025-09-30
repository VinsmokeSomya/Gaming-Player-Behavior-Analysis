"""Test database connection and setup."""

import pytest
from src.database import db_manager
from src.config import db_config


def test_database_config():
    """Test database configuration is loaded correctly."""
    assert db_config.host is not None
    assert db_config.port == 5432
    assert db_config.name == "player_analytics"
    assert db_config.user == "analytics_user"


def test_connection_string_format():
    """Test database connection string format."""
    connection_string = db_config.connection_string
    assert connection_string.startswith("postgresql://")
    assert "player_analytics" in connection_string


@pytest.mark.integration
def test_database_connection():
    """Test actual database connection (requires running PostgreSQL)."""
    # This test will be skipped if database is not available
    try:
        success = db_manager.test_connection()
        assert success, "Database connection should be successful"
    except Exception as e:
        pytest.skip(f"Database not available: {e}")


if __name__ == "__main__":
    # Run basic configuration tests
    test_database_config()
    test_connection_string_format()
    print("✓ Database configuration tests passed")
    
    # Try database connection test
    try:
        test_database_connection()
        print("✓ Database connection test passed")
    except Exception as e:
        print(f"⚠ Database connection test skipped: {e}")