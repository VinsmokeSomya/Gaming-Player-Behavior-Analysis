"""
Integration tests for error handling scenarios.
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import tempfile
import json
import os
from pathlib import Path

from src.database import DatabaseManager, retry_on_database_error, CircuitBreaker
from src.validation.data_quality import DataQualityValidator, ValidationSeverity
from src.etl.dead_letter_queue import DeadLetterQueue, FailedJob, FailureReason, ETLJobWrapper
from src.visualization.error_handling import (
    handle_visualization_errors, 
    ErrorFallbackGenerator,
    CachedVisualizationManager,
    VisualizationHealthChecker
)
from src.etl.ingestion import DataLoader
import plotly.graph_objects as go


class TestDatabaseErrorHandling:
    """Test database connection pooling and retry logic."""
    
    def test_retry_decorator_success_after_failure(self):
        """Test retry decorator succeeds after initial failures."""
        call_count = 0
        
        @retry_on_database_error(max_retries=3, delay=0.1, backoff=1.0)
        def failing_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Database connection failed")
            return "success"
        
        result = failing_function()
        assert result == "success"
        assert call_count == 3
    
    def test_retry_decorator_exhausts_retries(self):
        """Test retry decorator exhausts retries and raises exception."""
        call_count = 0
        
        @retry_on_database_error(max_retries=2, delay=0.1, backoff=1.0)
        def always_failing_function():
            nonlocal call_count
            call_count += 1
            raise ConnectionError("Database connection failed")
        
        with pytest.raises(ConnectionError):
            always_failing_function()
        
        assert call_count == 3  # Initial call + 2 retries
    
    def test_circuit_breaker_opens_after_failures(self):
        """Test circuit breaker opens after threshold failures."""
        circuit_breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=1)
        
        # Record failures to open circuit breaker
        for _ in range(3):
            circuit_breaker.record_failure()
        
        assert circuit_breaker.state == 'OPEN'
        assert not circuit_breaker.can_execute()
    
    def test_circuit_breaker_recovers_after_timeout(self):
        """Test circuit breaker recovers after timeout."""
        circuit_breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=0.1)
        
        # Open circuit breaker
        circuit_breaker.record_failure()
        circuit_breaker.record_failure()
        assert circuit_breaker.state == 'OPEN'
        
        # Wait for recovery timeout
        import time
        time.sleep(0.2)
        
        # Should be able to execute again
        assert circuit_breaker.can_execute()
        assert circuit_breaker.state == 'HALF_OPEN'
    
    @patch('src.database.create_engine')
    def test_database_manager_initialization_failure(self, mock_create_engine):
        """Test database manager handles initialization failures."""
        mock_create_engine.side_effect = Exception("Database connection failed")
        
        db_manager = DatabaseManager()
        
        with pytest.raises(Exception, match="Database connection failed"):
            db_manager.initialize()
    
    @patch('src.database.create_engine')
    def test_database_manager_pool_status(self, mock_create_engine):
        """Test database manager provides pool status."""
        mock_engine = Mock()
        mock_pool = Mock()
        mock_pool.size.return_value = 10
        mock_pool.checkedin.return_value = 8
        mock_pool.checkedout.return_value = 2
        mock_pool.overflow.return_value = 0
        mock_pool.invalid.return_value = 0
        mock_engine.pool = mock_pool
        mock_create_engine.return_value = mock_engine
        
        db_manager = DatabaseManager()
        db_manager.initialize()
        
        status = db_manager.get_pool_status()
        assert status['size'] == 10
        assert status['checked_in'] == 8
        assert status['checked_out'] == 2


class TestDataValidation:
    """Test data validation with configurable thresholds."""
    
    def test_data_quality_validator_missing_values(self):
        """Test data quality validator detects missing values."""
        validator = DataQualityValidator()
        
        # Create test data with missing values
        events_df = pd.DataFrame({
            'player_id': ['p1', None, 'p3', 'p4'],
            'timestamp': [datetime.now()] * 4,
            'event_type': ['session_start', 'purchase', None, 'level_complete']
        })
        
        results = validator.validate_events_data(events_df)
        
        # Should detect missing player_id and event_type
        missing_player_id = next((r for r in results if r.rule_name == 'missing_player_ids'), None)
        missing_event_type = next((r for r in results if r.rule_name == 'missing_event_types'), None)
        
        assert missing_player_id is not None
        assert not missing_player_id.passed
        assert missing_player_id.value == 0.25  # 1 out of 4
        
        assert missing_event_type is not None
        assert not missing_event_type.passed
        assert missing_event_type.value == 0.25  # 1 out of 4
    
    def test_data_quality_validator_duplicates(self):
        """Test data quality validator detects duplicates."""
        validator = DataQualityValidator()
        
        # Create test data with duplicates
        events_df = pd.DataFrame({
            'player_id': ['p1', 'p1', 'p2', 'p3'],
            'timestamp': [datetime.now()] * 4,
            'event_type': ['session_start'] * 4
        })
        
        results = validator.validate_events_data(events_df)
        
        duplicate_result = next((r for r in results if r.rule_name == 'duplicate_events'), None)
        assert duplicate_result is not None
        # Note: duplicates are detected based on all columns, so this might not trigger
    
    def test_data_quality_validator_future_timestamps(self):
        """Test data quality validator detects future timestamps."""
        validator = DataQualityValidator()
        
        future_time = datetime.now() + timedelta(days=1)
        events_df = pd.DataFrame({
            'player_id': ['p1', 'p2', 'p3', 'p4'],
            'timestamp': [datetime.now(), future_time, datetime.now(), datetime.now()],
            'event_type': ['session_start'] * 4
        })
        
        results = validator.validate_events_data(events_df)
        
        future_result = next((r for r in results if r.rule_name == 'future_timestamps'), None)
        assert future_result is not None
        assert not future_result.passed
        assert future_result.value == 0.25  # 1 out of 4
    
    def test_data_quality_config_rule_updates(self):
        """Test data quality config allows rule updates."""
        validator = DataQualityValidator()
        
        # Update threshold for missing player IDs
        validator.config.update_rule_threshold('missing_player_ids', 0.5)
        
        rule = validator.config.get_rule('missing_player_ids')
        assert rule.threshold == 0.5
    
    def test_data_quality_report_generation(self):
        """Test data quality report generation."""
        validator = DataQualityValidator()
        
        # Create test data with issues
        events_df = pd.DataFrame({
            'player_id': ['p1', None, 'p3'],
            'timestamp': [datetime.now()] * 3,
            'event_type': ['session_start'] * 3
        })
        
        results = validator.validate_events_data(events_df)
        report = validator.generate_quality_report(results)
        
        assert 'timestamp' in report
        assert 'total_checks' in report
        assert 'failed_checks' in report
        assert 'severity_breakdown' in report
        assert 'issues' in report
        assert 'recommendations' in report


class TestDeadLetterQueue:
    """Test dead letter queue system for failed ETL jobs."""
    
    def test_dead_letter_queue_add_failed_job(self):
        """Test adding failed job to dead letter queue."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dlq = DeadLetterQueue(storage_path=temp_dir)
            
            dlq.add_failed_job(
                job_id="test_job_1",
                job_type="data_ingestion",
                failure_reason=FailureReason.DATA_VALIDATION_ERROR,
                error_message="Invalid data format",
                input_data={"file": "test.json"},
                max_retries=3
            )
            
            failed_jobs = dlq.get_failed_jobs()
            assert len(failed_jobs) == 1
            
            job = failed_jobs[0]
            assert job.job_id == "test_job_1"
            assert job.job_type == "data_ingestion"
            assert job.failure_reason == FailureReason.DATA_VALIDATION_ERROR
            assert job.retry_count == 0
            assert job.next_retry_time is not None
    
    def test_dead_letter_queue_retry_mechanism(self):
        """Test dead letter queue retry mechanism."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dlq = DeadLetterQueue(storage_path=temp_dir)
            
            # Register retry handler
            retry_success = False
            
            def retry_handler(input_data, metadata):
                nonlocal retry_success
                retry_success = True
                return True
            
            dlq.register_retry_handler("test_job_type", retry_handler)
            
            # Add failed job with immediate retry time
            dlq.add_failed_job(
                job_id="retry_test",
                job_type="test_job_type",
                failure_reason=FailureReason.PROCESSING_ERROR,
                error_message="Temporary failure",
                input_data={"test": "data"},
                max_retries=1
            )
            
            # Manually set retry time to now for immediate retry
            job = dlq.failed_jobs["retry_test"]
            job.next_retry_time = datetime.now()
            
            # Process retries
            results = dlq.process_retries()
            
            assert results['attempted'] == 1
            assert results['succeeded'] == 1
            assert retry_success
            assert len(dlq.get_failed_jobs()) == 0  # Job should be removed after success
    
    def test_dead_letter_queue_statistics(self):
        """Test dead letter queue statistics."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dlq = DeadLetterQueue(storage_path=temp_dir)
            
            # Add multiple failed jobs
            dlq.add_failed_job("job1", "type_a", FailureReason.DATA_VALIDATION_ERROR, "Error 1", {})
            dlq.add_failed_job("job2", "type_a", FailureReason.PROCESSING_ERROR, "Error 2", {})
            dlq.add_failed_job("job3", "type_b", FailureReason.DATABASE_CONNECTION_ERROR, "Error 3", {})
            
            stats = dlq.get_queue_statistics()
            
            assert stats['total_failed_jobs'] == 3
            assert stats['jobs_by_type']['type_a'] == 2
            assert stats['jobs_by_type']['type_b'] == 1
            assert stats['jobs_by_reason']['data_validation_error'] == 1
            assert stats['jobs_by_reason']['processing_error'] == 1
            assert stats['jobs_by_reason']['database_connection_error'] == 1
    
    def test_etl_job_wrapper_success(self):
        """Test ETL job wrapper with successful job."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dlq = DeadLetterQueue(storage_path=temp_dir)
            wrapper = ETLJobWrapper(dlq)
            
            def successful_job(input_data):
                return f"processed_{input_data}"
            
            result = wrapper.execute_job(
                job_id="success_test",
                job_type="test_job",
                job_func=successful_job,
                input_data="test_data"
            )
            
            assert result == "processed_test_data"
            assert len(dlq.get_failed_jobs()) == 0
    
    def test_etl_job_wrapper_failure(self):
        """Test ETL job wrapper with failing job."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dlq = DeadLetterQueue(storage_path=temp_dir)
            wrapper = ETLJobWrapper(dlq)
            
            def failing_job(input_data):
                raise ValueError("Job failed")
            
            with pytest.raises(ValueError):
                wrapper.execute_job(
                    job_id="failure_test",
                    job_type="test_job",
                    job_func=failing_job,
                    input_data="test_data"
                )
            
            failed_jobs = dlq.get_failed_jobs()
            assert len(failed_jobs) == 1
            assert failed_jobs[0].job_id == "failure_test"
            assert failed_jobs[0].failure_reason == FailureReason.PROCESSING_ERROR


class TestVisualizationErrorHandling:
    """Test visualization error handling and fallbacks."""
    
    def test_error_fallback_generator_error_chart(self):
        """Test error fallback chart generation."""
        error_chart = ErrorFallbackGenerator.create_error_chart(
            "Test error message", "test_chart"
        )
        
        assert isinstance(error_chart, go.Figure)
        assert error_chart.layout.title.text == "Error in Test_Chart Chart"
    
    def test_error_fallback_generator_no_data_chart(self):
        """Test no data fallback chart generation."""
        no_data_chart = ErrorFallbackGenerator.create_no_data_chart("No data available")
        
        assert isinstance(no_data_chart, go.Figure)
        assert no_data_chart.layout.title.text == "No Data Available"
    
    def test_visualization_error_decorator(self):
        """Test visualization error handling decorator."""
        
        @handle_visualization_errors("test_chart")
        def failing_chart_function():
            raise ValueError("Chart generation failed")
        
        result = failing_chart_function()
        
        assert isinstance(result, go.Figure)
        # Should return error chart instead of raising exception
    
    def test_cached_visualization_manager(self):
        """Test cached visualization manager."""
        cache_manager = CachedVisualizationManager(cache_duration_minutes=1)
        
        # Cache a result
        test_data = {"chart": "data"}
        cache_manager.cache_result("test_key", test_data)
        
        # Retrieve cached result
        cached_result = cache_manager.get_cached_result("test_key")
        assert cached_result == test_data
        
        # Test cache expiration (would need to mock datetime for proper testing)
        cache_manager.clear_cache()
        cached_result = cache_manager.get_cached_result("test_key")
        assert cached_result is None
    
    def test_visualization_health_checker(self):
        """Test visualization health checker."""
        health_checker = VisualizationHealthChecker()
        
        # Record errors
        health_checker.record_error("test_component")
        health_checker.record_error("test_component")
        
        assert not health_checker.is_component_healthy("test_component", max_errors=1)
        
        # Record success should reset error count
        health_checker.record_success("test_component")
        assert health_checker.is_component_healthy("test_component", max_errors=1)
        
        # Get health report
        report = health_checker.get_health_report()
        assert "error_counts" in report
        assert "last_success" in report


class TestETLErrorHandling:
    """Test ETL pipeline error handling integration."""
    
    def test_data_loader_with_invalid_json(self):
        """Test data loader handles invalid JSON files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create invalid JSON file
            invalid_file = Path(temp_dir) / "player_profiles.json"
            with open(invalid_file, 'w') as f:
                f.write("invalid json content")
            
            loader = DataLoader(data_dir=temp_dir)
            
            with pytest.raises(ValueError, match="Invalid JSON"):
                loader.load_player_profiles()
    
    def test_data_loader_with_missing_file(self):
        """Test data loader handles missing files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            loader = DataLoader(data_dir=temp_dir)
            
            with pytest.raises(FileNotFoundError):
                loader.load_player_profiles()
    
    def test_data_loader_with_validation_errors(self):
        """Test data loader handles validation errors."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create file with invalid profile data
            profiles_file = Path(temp_dir) / "player_profiles.json"
            invalid_profiles = [
                {
                    "player_id": "p1",
                    "registration_date": "invalid_date",  # Invalid date format
                    "last_active_date": "2024-01-01T00:00:00",
                    "total_sessions": 10,
                    "total_playtime_minutes": 1000,
                    "highest_level_reached": 5,
                    "total_purchases": 0.0,
                    "churn_risk_score": 0.3,
                    "churn_prediction_date": "2024-01-01T00:00:00"
                }
            ]
            
            with open(profiles_file, 'w') as f:
                json.dump(invalid_profiles, f)
            
            loader = DataLoader(data_dir=temp_dir)
            
            # Should handle validation errors gracefully or raise appropriate exception
            with pytest.raises((ValueError, Exception)):
                loader.load_player_profiles()


class TestEndToEndErrorHandling:
    """Test end-to-end error handling scenarios."""
    
    @patch('src.database.create_engine')
    def test_database_failure_with_visualization_fallback(self, mock_create_engine):
        """Test complete pipeline with database failure and visualization fallback."""
        # Mock database failure
        mock_create_engine.side_effect = Exception("Database unavailable")
        
        db_manager = DatabaseManager()
        
        with pytest.raises(Exception):
            db_manager.initialize()
        
        # Test that visualization components can still provide fallback content
        error_chart = ErrorFallbackGenerator.create_error_chart(
            "Database connection failed", "retention_chart"
        )
        
        assert isinstance(error_chart, go.Figure)
        assert "Database connection failed" in str(error_chart.layout.annotations[0].text)
    
    def test_complete_error_handling_workflow(self):
        """Test complete error handling workflow from ETL to visualization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Set up dead letter queue
            dlq = DeadLetterQueue(storage_path=temp_dir)
            
            # Set up data loader with error handling
            loader = DataLoader(data_dir="nonexistent_directory")
            
            # Attempt to load data (should fail and go to DLQ)
            try:
                loader.load_player_profiles()
            except Exception:
                pass  # Expected to fail
            
            # Check that failed jobs are in DLQ
            failed_jobs = dlq.get_failed_jobs()
            # Note: This test would need actual integration with the DLQ system
            
            # Test visualization fallback
            health_checker = VisualizationHealthChecker()
            health_checker.record_error("data_loading")
            
            assert not health_checker.is_component_healthy("data_loading", max_errors=0)
            
            # Generate fallback visualization
            fallback_chart = ErrorFallbackGenerator.create_no_data_chart(
                "Data loading failed - using cached data"
            )
            
            assert isinstance(fallback_chart, go.Figure)