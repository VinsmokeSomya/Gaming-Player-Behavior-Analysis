"""
Dead letter queue system for failed ETL jobs.
"""

import logging
import json
import pickle
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import os
from pathlib import Path
import threading
import time

logger = logging.getLogger(__name__)


class FailureReason(Enum):
    """Reasons for ETL job failures."""
    DATA_VALIDATION_ERROR = "data_validation_error"
    DATABASE_CONNECTION_ERROR = "database_connection_error"
    PROCESSING_ERROR = "processing_error"
    TIMEOUT_ERROR = "timeout_error"
    MEMORY_ERROR = "memory_error"
    UNKNOWN_ERROR = "unknown_error"


@dataclass
class FailedJob:
    """Represents a failed ETL job."""
    job_id: str
    job_type: str
    failure_reason: FailureReason
    error_message: str
    input_data: Any
    failure_timestamp: datetime
    retry_count: int = 0
    max_retries: int = 3
    next_retry_time: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['failure_reason'] = self.failure_reason.value
        data['failure_timestamp'] = self.failure_timestamp.isoformat()
        if self.next_retry_time:
            data['next_retry_time'] = self.next_retry_time.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FailedJob':
        """Create from dictionary."""
        data['failure_reason'] = FailureReason(data['failure_reason'])
        data['failure_timestamp'] = datetime.fromisoformat(data['failure_timestamp'])
        if data.get('next_retry_time'):
            data['next_retry_time'] = datetime.fromisoformat(data['next_retry_time'])
        return cls(**data)


class DeadLetterQueue:
    """Dead letter queue for managing failed ETL jobs."""
    
    def __init__(self, storage_path: str = "data/dlq", max_queue_size: int = 1000):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.max_queue_size = max_queue_size
        self.failed_jobs: Dict[str, FailedJob] = {}
        self.retry_handlers: Dict[str, Callable] = {}
        self._lock = threading.Lock()
        
        # Load existing failed jobs
        self._load_failed_jobs()
    
    def add_failed_job(self, job_id: str, job_type: str, failure_reason: FailureReason,
                      error_message: str, input_data: Any, max_retries: int = 3,
                      metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add a failed job to the dead letter queue."""
        with self._lock:
            # Serialize input data to handle non-JSON serializable types
            serialized_input_data = self._serialize_for_storage(input_data)
            
            failed_job = FailedJob(
                job_id=job_id,
                job_type=job_type,
                failure_reason=failure_reason,
                error_message=error_message,
                input_data=serialized_input_data,
                failure_timestamp=datetime.now(),
                max_retries=max_retries,
                metadata=metadata or {}
            )
            
            # Calculate next retry time with exponential backoff
            if failed_job.retry_count < max_retries:
                backoff_minutes = 2 ** failed_job.retry_count  # 1, 2, 4, 8 minutes
                failed_job.next_retry_time = datetime.now() + timedelta(minutes=backoff_minutes)
            
            self.failed_jobs[job_id] = failed_job
            self._persist_failed_job(failed_job)
            
            logger.error(f"Added failed job to DLQ: {job_id} - {error_message}")
            
            # Cleanup old jobs if queue is too large
            self._cleanup_old_jobs()
    
    def _serialize_for_storage(self, data: Any) -> Any:
        """Serialize data for JSON storage."""
        try:
            if hasattr(data, '__fspath__'):  # Path-like object
                return str(data)
            elif hasattr(data, 'to_dict'):  # Custom objects with to_dict method
                return data.to_dict()
            elif isinstance(data, (list, tuple)):
                return [self._serialize_for_storage(item) for item in data]
            elif isinstance(data, dict):
                return {k: self._serialize_for_storage(v) for k, v in data.items()}
            else:
                # Try JSON serialization test
                json.dumps(data)
                return data
        except (TypeError, ValueError):
            # Fallback to string representation
            return str(data)
    
    def register_retry_handler(self, job_type: str, handler: Callable) -> None:
        """Register a retry handler for a specific job type."""
        self.retry_handlers[job_type] = handler
        logger.info(f"Registered retry handler for job type: {job_type}")
    
    def process_retries(self) -> Dict[str, Any]:
        """Process jobs that are ready for retry."""
        retry_results = {
            'attempted': 0,
            'succeeded': 0,
            'failed': 0,
            'details': []
        }
        
        now = datetime.now()
        jobs_to_retry = []
        
        with self._lock:
            for job_id, failed_job in self.failed_jobs.items():
                if (failed_job.next_retry_time and 
                    failed_job.next_retry_time <= now and
                    failed_job.retry_count < failed_job.max_retries):
                    jobs_to_retry.append(failed_job)
        
        for failed_job in jobs_to_retry:
            retry_results['attempted'] += 1
            
            try:
                success = self._retry_job(failed_job)
                
                if success:
                    retry_results['succeeded'] += 1
                    self._remove_failed_job(failed_job.job_id)
                    retry_results['details'].append({
                        'job_id': failed_job.job_id,
                        'status': 'succeeded',
                        'retry_count': failed_job.retry_count + 1
                    })
                else:
                    retry_results['failed'] += 1
                    self._update_failed_job_retry(failed_job)
                    retry_results['details'].append({
                        'job_id': failed_job.job_id,
                        'status': 'failed',
                        'retry_count': failed_job.retry_count + 1
                    })
                    
            except Exception as e:
                retry_results['failed'] += 1
                logger.error(f"Error during retry of job {failed_job.job_id}: {e}")
                self._update_failed_job_retry(failed_job)
                retry_results['details'].append({
                    'job_id': failed_job.job_id,
                    'status': 'error',
                    'error': str(e),
                    'retry_count': failed_job.retry_count + 1
                })
        
        if retry_results['attempted'] > 0:
            logger.info(f"Processed {retry_results['attempted']} retry attempts: "
                       f"{retry_results['succeeded']} succeeded, {retry_results['failed']} failed")
        
        return retry_results
    
    def get_failed_jobs(self, job_type: Optional[str] = None, 
                       failure_reason: Optional[FailureReason] = None) -> List[FailedJob]:
        """Get failed jobs with optional filtering."""
        with self._lock:
            jobs = list(self.failed_jobs.values())
        
        if job_type:
            jobs = [job for job in jobs if job.job_type == job_type]
        
        if failure_reason:
            jobs = [job for job in jobs if job.failure_reason == failure_reason]
        
        return sorted(jobs, key=lambda x: x.failure_timestamp, reverse=True)
    
    def get_queue_statistics(self) -> Dict[str, Any]:
        """Get statistics about the dead letter queue."""
        with self._lock:
            jobs = list(self.failed_jobs.values())
        
        stats = {
            'total_failed_jobs': len(jobs),
            'jobs_by_type': {},
            'jobs_by_reason': {},
            'jobs_ready_for_retry': 0,
            'jobs_exhausted_retries': 0,
            'oldest_job': None,
            'newest_job': None
        }
        
        if jobs:
            # Count by type and reason
            for job in jobs:
                stats['jobs_by_type'][job.job_type] = stats['jobs_by_type'].get(job.job_type, 0) + 1
                reason_key = job.failure_reason.value
                stats['jobs_by_reason'][reason_key] = stats['jobs_by_reason'].get(reason_key, 0) + 1
                
                # Count retry status
                if job.retry_count >= job.max_retries:
                    stats['jobs_exhausted_retries'] += 1
                elif job.next_retry_time and job.next_retry_time <= datetime.now():
                    stats['jobs_ready_for_retry'] += 1
            
            # Find oldest and newest
            jobs_sorted = sorted(jobs, key=lambda x: x.failure_timestamp)
            stats['oldest_job'] = jobs_sorted[0].failure_timestamp.isoformat()
            stats['newest_job'] = jobs_sorted[-1].failure_timestamp.isoformat()
        
        return stats
    
    def remove_job(self, job_id: str) -> bool:
        """Manually remove a job from the dead letter queue."""
        return self._remove_failed_job(job_id)
    
    def clear_old_jobs(self, older_than_days: int = 7) -> int:
        """Clear jobs older than specified days."""
        cutoff_date = datetime.now() - timedelta(days=older_than_days)
        removed_count = 0
        
        with self._lock:
            jobs_to_remove = [
                job_id for job_id, job in self.failed_jobs.items()
                if job.failure_timestamp < cutoff_date
            ]
        
        for job_id in jobs_to_remove:
            if self._remove_failed_job(job_id):
                removed_count += 1
        
        logger.info(f"Removed {removed_count} jobs older than {older_than_days} days")
        return removed_count
    
    def _retry_job(self, failed_job: FailedJob) -> bool:
        """Attempt to retry a failed job."""
        handler = self.retry_handlers.get(failed_job.job_type)
        if not handler:
            logger.warning(f"No retry handler registered for job type: {failed_job.job_type}")
            return False
        
        try:
            logger.info(f"Retrying job {failed_job.job_id} (attempt {failed_job.retry_count + 1})")
            result = handler(failed_job.input_data, failed_job.metadata)
            return result is not False  # Consider None as success
        except Exception as e:
            logger.error(f"Retry failed for job {failed_job.job_id}: {e}")
            return False
    
    def _update_failed_job_retry(self, failed_job: FailedJob) -> None:
        """Update failed job after retry attempt."""
        with self._lock:
            failed_job.retry_count += 1
            
            if failed_job.retry_count < failed_job.max_retries:
                # Calculate next retry time with exponential backoff
                backoff_minutes = 2 ** failed_job.retry_count
                failed_job.next_retry_time = datetime.now() + timedelta(minutes=backoff_minutes)
            else:
                failed_job.next_retry_time = None  # No more retries
            
            self.failed_jobs[failed_job.job_id] = failed_job
            self._persist_failed_job(failed_job)
    
    def _remove_failed_job(self, job_id: str) -> bool:
        """Remove a failed job from the queue."""
        with self._lock:
            if job_id in self.failed_jobs:
                del self.failed_jobs[job_id]
                
                # Remove from persistent storage
                job_file = self.storage_path / f"{job_id}.json"
                if job_file.exists():
                    job_file.unlink()
                
                logger.info(f"Removed job {job_id} from dead letter queue")
                return True
        return False
    
    def _persist_failed_job(self, failed_job: FailedJob) -> None:
        """Persist failed job to storage."""
        try:
            job_file = self.storage_path / f"{failed_job.job_id}.json"
            with open(job_file, 'w') as f:
                json.dump(failed_job.to_dict(), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to persist job {failed_job.job_id}: {e}")
    
    def _load_failed_jobs(self) -> None:
        """Load failed jobs from persistent storage."""
        try:
            for job_file in self.storage_path.glob("*.json"):
                try:
                    with open(job_file, 'r') as f:
                        job_data = json.load(f)
                    
                    failed_job = FailedJob.from_dict(job_data)
                    self.failed_jobs[failed_job.job_id] = failed_job
                    
                except Exception as e:
                    logger.error(f"Failed to load job from {job_file}: {e}")
            
            logger.info(f"Loaded {len(self.failed_jobs)} failed jobs from storage")
            
        except Exception as e:
            logger.error(f"Failed to load failed jobs: {e}")
    
    def _cleanup_old_jobs(self) -> None:
        """Clean up old jobs if queue exceeds maximum size."""
        if len(self.failed_jobs) > self.max_queue_size:
            # Remove oldest jobs that have exhausted retries
            exhausted_jobs = [
                (job_id, job) for job_id, job in self.failed_jobs.items()
                if job.retry_count >= job.max_retries
            ]
            
            # Sort by timestamp and remove oldest
            exhausted_jobs.sort(key=lambda x: x[1].failure_timestamp)
            jobs_to_remove = len(self.failed_jobs) - self.max_queue_size
            
            for i in range(min(jobs_to_remove, len(exhausted_jobs))):
                job_id = exhausted_jobs[i][0]
                self._remove_failed_job(job_id)


class ETLJobWrapper:
    """Wrapper for ETL jobs with automatic dead letter queue integration."""
    
    def __init__(self, dlq: DeadLetterQueue):
        self.dlq = dlq
    
    def execute_job(self, job_id: str, job_type: str, job_func: Callable, 
                   input_data: Any, timeout_seconds: int = 300,
                   metadata: Optional[Dict[str, Any]] = None) -> Any:
        """Execute an ETL job with automatic error handling."""
        start_time = time.time()
        
        try:
            # Set up timeout if specified (Windows compatible)
            timeout_thread = None
            if timeout_seconds > 0:
                import threading
                
                def timeout_handler():
                    time.sleep(timeout_seconds)
                    raise TimeoutError(f"Job {job_id} timed out after {timeout_seconds} seconds")
                
                # Note: This is a simplified timeout mechanism for Windows compatibility
                # In production, consider using more robust timeout mechanisms
                pass  # Skip timeout for now to avoid Windows signal issues
            
            # Execute the job
            result = job_func(input_data)
            
            execution_time = time.time() - start_time
            logger.info(f"Job {job_id} completed successfully in {execution_time:.2f}s")
            
            return result
            
        except TimeoutError as e:
            self.dlq.add_failed_job(
                job_id=job_id,
                job_type=job_type,
                failure_reason=FailureReason.TIMEOUT_ERROR,
                error_message=str(e),
                input_data=self._serialize_input_data(input_data),
                metadata=metadata
            )
            raise
            
        except MemoryError as e:
            self.dlq.add_failed_job(
                job_id=job_id,
                job_type=job_type,
                failure_reason=FailureReason.MEMORY_ERROR,
                error_message=str(e),
                input_data=self._serialize_input_data(input_data),
                metadata=metadata
            )
            raise
            
        except Exception as e:
            # Determine failure reason based on error type
            if "validation" in str(e).lower():
                failure_reason = FailureReason.DATA_VALIDATION_ERROR
            elif "connection" in str(e).lower() or "database" in str(e).lower():
                failure_reason = FailureReason.DATABASE_CONNECTION_ERROR
            else:
                failure_reason = FailureReason.PROCESSING_ERROR
            
            self.dlq.add_failed_job(
                job_id=job_id,
                job_type=job_type,
                failure_reason=failure_reason,
                error_message=str(e),
                input_data=self._serialize_input_data(input_data),
                metadata=metadata
            )
            raise
    
    def _serialize_input_data(self, input_data: Any) -> Any:
        """Serialize input data for storage, handling non-serializable types."""
        try:
            # Try to convert Path objects to strings
            if hasattr(input_data, '__fspath__'):  # Path-like object
                return str(input_data)
            return input_data
        except Exception:
            return str(input_data)


# Global dead letter queue instance
dlq = DeadLetterQueue()