"""Database connection utilities and session management."""

import logging
from contextlib import contextmanager
from typing import Generator, Optional
from sqlalchemy import create_engine, Engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from sqlalchemy.exc import SQLAlchemyError
import time

from .config import db_config, app_config

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages database connections and sessions with connection pooling."""
    
    def __init__(self):
        self._engine: Optional[Engine] = None
        self._session_factory: Optional[sessionmaker] = None
    
    def initialize(self) -> None:
        """Initialize database engine and session factory."""
        try:
            self._engine = create_engine(
                db_config.connection_string,
                poolclass=QueuePool,
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True,
                pool_recycle=3600,
                echo=app_config.log_level == "DEBUG"
            )
            
            self._session_factory = sessionmaker(
                bind=self._engine,
                autocommit=False,
                autoflush=False
            )
            
            logger.info("Database connection initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database connection: {e}")
            raise
    
    @property
    def engine(self) -> Engine:
        """Get database engine."""
        if self._engine is None:
            self.initialize()
        return self._engine
    
    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """Get database session with automatic cleanup."""
        if self._session_factory is None:
            self.initialize()
        
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()
    
    def test_connection(self) -> bool:
        """Test database connection."""
        try:
            with self.get_session() as session:
                session.execute(text("SELECT 1"))
            logger.info("Database connection test successful")
            return True
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False
    
    def wait_for_connection(self, max_retries: int = 30, retry_interval: int = 2) -> bool:
        """Wait for database to become available."""
        for attempt in range(max_retries):
            try:
                if self.test_connection():
                    return True
            except Exception as e:
                logger.warning(f"Connection attempt {attempt + 1} failed: {e}")
            
            if attempt < max_retries - 1:
                time.sleep(retry_interval)
        
        logger.error(f"Failed to connect to database after {max_retries} attempts")
        return False
    
    def execute_query(self, query: str, params: Optional[dict] = None) -> list:
        """Execute a query and return results."""
        try:
            with self.get_session() as session:
                result = session.execute(text(query), params or {})
                return result.fetchall()
        except SQLAlchemyError as e:
            logger.error(f"Query execution failed: {e}")
            raise
    
    def close(self) -> None:
        """Close database connections."""
        if self._engine:
            self._engine.dispose()
            logger.info("Database connections closed")


# Global database manager instance
db_manager = DatabaseManager()