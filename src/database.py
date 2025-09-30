"""Database connection utilities and session management."""

import logging
from contextlib import contextmanager
from typing import Generator, Optional, Any, Dict, List
from sqlalchemy import create_engine, Engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from sqlalchemy.exc import SQLAlchemyError, DisconnectionError, OperationalError
import time
import functools
from datetime import datetime, timedelta

from .config import db_config, app_config

logger = logging.getLogger(__name__)


class CircuitBreaker:
    """Circuit breaker pattern for database operations."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
    
    def can_execute(self) -> bool:
        """Check if operation can be executed based on circuit breaker state."""
        if self.state == 'CLOSED':
            return True
        elif self.state == 'OPEN':
            if self.last_failure_time and \
               (datetime.now() - self.last_failure_time).total_seconds() > self.recovery_timeout:
                self.state = 'HALF_OPEN'
                return True
            return False
        elif self.state == 'HALF_OPEN':
            return True
        return False
    
    def record_success(self):
        """Record successful operation."""
        self.failure_count = 0
        self.state = 'CLOSED'
    
    def record_failure(self):
        """Record failed operation."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.failure_threshold:
            self.state = 'OPEN'
            logger.warning(f"Circuit breaker opened after {self.failure_count} failures")


def retry_on_database_error(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """Decorator for retrying database operations on transient failures."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except (DisconnectionError, OperationalError, ConnectionError) as e:
                    last_exception = e
                    if attempt == max_retries:
                        logger.error(f"Database operation failed after {max_retries} retries: {e}")
                        raise
                    
                    wait_time = delay * (backoff ** attempt)
                    logger.warning(f"Database operation failed (attempt {attempt + 1}/{max_retries + 1}), retrying in {wait_time}s: {e}")
                    time.sleep(wait_time)
                except Exception as e:
                    # Don't retry on non-transient errors
                    logger.error(f"Non-retryable database error: {e}")
                    raise
            
            # This should never be reached, but just in case
            raise last_exception
        return wrapper
    return decorator


class ConnectionPool:
    """Enhanced connection pool with monitoring and health checks."""
    
    def __init__(self, engine: Engine):
        self.engine = engine
        self.pool_stats = {
            'total_connections': 0,
            'active_connections': 0,
            'failed_connections': 0,
            'last_health_check': None
        }
    
    def get_pool_status(self) -> Dict[str, Any]:
        """Get current pool status and statistics."""
        pool = self.engine.pool
        return {
            'size': pool.size(),
            'checked_in': pool.checkedin(),
            'checked_out': pool.checkedout(),
            'overflow': pool.overflow(),
            'invalid': pool.invalid(),
            'stats': self.pool_stats.copy()
        }
    
    def health_check(self) -> bool:
        """Perform health check on connection pool."""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            self.pool_stats['last_health_check'] = datetime.now()
            return True
        except Exception as e:
            logger.error(f"Connection pool health check failed: {e}")
            self.pool_stats['failed_connections'] += 1
            return False


class DatabaseManager:
    """Manages database connections and sessions with enhanced connection pooling and error handling."""
    
    def __init__(self):
        self._engine: Optional[Engine] = None
        self._session_factory: Optional[sessionmaker] = None
        self._connection_pool: Optional[ConnectionPool] = None
        self._circuit_breaker = CircuitBreaker()
    
    def initialize(self) -> None:
        """Initialize database engine and session factory with enhanced configuration."""
        try:
            # Enhanced connection pool configuration
            self._engine = create_engine(
                db_config.connection_string,
                poolclass=QueuePool,
                pool_size=15,  # Increased pool size
                max_overflow=30,  # Increased overflow
                pool_pre_ping=True,
                pool_recycle=3600,
                pool_timeout=30,  # Connection timeout
                pool_reset_on_return='commit',  # Reset connections on return
                echo=app_config.log_level == "DEBUG",
                # Connection arguments for better reliability
                connect_args={
                    "connect_timeout": 10,
                    "application_name": "player_analytics",
                    "options": "-c statement_timeout=30000"  # 30 second statement timeout
                }
            )
            
            self._session_factory = sessionmaker(
                bind=self._engine,
                autocommit=False,
                autoflush=False,
                expire_on_commit=False  # Prevent lazy loading issues
            )
            
            # Initialize connection pool monitoring
            self._connection_pool = ConnectionPool(self._engine)
            
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
    @retry_on_database_error(max_retries=3, delay=1.0, backoff=2.0)
    def get_session(self) -> Generator[Session, None, None]:
        """Get database session with automatic cleanup and retry logic."""
        if self._session_factory is None:
            self.initialize()
        
        if not self._circuit_breaker.can_execute():
            raise Exception("Database circuit breaker is open - too many recent failures")
        
        session = self._session_factory()
        try:
            yield session
            session.commit()
            self._circuit_breaker.record_success()
        except Exception as e:
            session.rollback()
            self._circuit_breaker.record_failure()
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
    
    @retry_on_database_error(max_retries=3, delay=0.5, backoff=1.5)
    def execute_query(self, query: str, params: Optional[dict] = None) -> list:
        """Execute a query and return results with retry logic."""
        try:
            with self.get_session() as session:
                result = session.execute(text(query), params or {})
                return result.fetchall()
        except SQLAlchemyError as e:
            logger.error(f"Query execution failed: {e}")
            raise
    
    def get_pool_status(self) -> Dict[str, Any]:
        """Get connection pool status and statistics."""
        if self._connection_pool is None:
            return {"status": "not_initialized"}
        return self._connection_pool.get_pool_status()
    
    def perform_health_check(self) -> bool:
        """Perform comprehensive database health check."""
        if self._connection_pool is None:
            return False
        return self._connection_pool.health_check()
    
    def close(self) -> None:
        """Close database connections."""
        if self._engine:
            self._engine.dispose()
            logger.info("Database connections closed")


# Global database manager instance
db_manager = DatabaseManager()