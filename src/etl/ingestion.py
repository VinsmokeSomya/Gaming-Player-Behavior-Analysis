"""
Data ingestion functions for processing raw player events and profiles.
"""
import json
import pandas as pd
from datetime import datetime, date
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import logging
import uuid

from ..models import PlayerProfile, ChurnFeatures, validate_player_event
from ..validation import DataQualityValidator, ValidationSeverity
from .dead_letter_queue import dlq, ETLJobWrapper, FailureReason

logger = logging.getLogger(__name__)


class DataLoader:
    """Loads and validates data from various sources with error handling."""
    
    def __init__(self, data_dir: str = "data/sample"):
        """Initialize data loader with data directory."""
        self.data_dir = Path(data_dir)
        self.validator = DataQualityValidator()
        self.job_wrapper = ETLJobWrapper(dlq)
        
    def load_player_profiles(self) -> List[PlayerProfile]:
        """Load and validate player profiles from JSON with error handling."""
        job_id = f"load_profiles_{uuid.uuid4().hex[:8]}"
        
        def _load_profiles_job(data_dir):
            profiles_file = data_dir / "player_profiles.json"
            
            if not profiles_file.exists():
                raise FileNotFoundError(f"Player profiles file not found: {profiles_file}")
            
            try:
                with open(profiles_file, 'r') as f:
                    profiles_data = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in profiles file: {e}")
            
            profiles = []
            validation_errors = []
            
            for i, profile_data in enumerate(profiles_data):
                try:
                    # Convert ISO strings back to datetime objects
                    for field in ['registration_date', 'last_active_date', 'churn_prediction_date']:
                        if isinstance(profile_data[field], str):
                            profile_data[field] = datetime.fromisoformat(profile_data[field])
                    
                    profile = PlayerProfile.from_dict(profile_data)
                    profiles.append(profile)
                    
                except Exception as e:
                    validation_errors.append(f"Profile {i}: {e}")
            
            # Check validation error threshold
            if validation_errors:
                error_rate = len(validation_errors) / len(profiles_data)
                if error_rate > 0.1:  # More than 10% errors
                    raise ValueError(f"Too many profile validation errors ({error_rate:.1%}): {validation_errors[:5]}")
                else:
                    logger.warning(f"Profile validation errors ({error_rate:.1%}): {validation_errors[:3]}")
            
            logger.info(f"Loaded {len(profiles)} player profiles with {len(validation_errors)} validation errors")
            return profiles
        
        return self.job_wrapper.execute_job(
            job_id=job_id,
            job_type="load_player_profiles",
            job_func=_load_profiles_job,
            input_data=self.data_dir,
            timeout_seconds=60
        )
    
    def load_player_events(self) -> List[Dict[str, Any]]:
        """Load and validate player events from JSON with error handling."""
        job_id = f"load_events_{uuid.uuid4().hex[:8]}"
        
        def _load_events_job(data_dir):
            events_file = data_dir / "player_events.json"
            
            if not events_file.exists():
                raise FileNotFoundError(f"Player events file not found: {events_file}")
            
            try:
                with open(events_file, 'r') as f:
                    events_data = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in events file: {e}")
            
            # Validate and convert events
            validated_events = []
            validation_errors = []
            
            for i, event in enumerate(events_data):
                try:
                    # Convert timestamp string to datetime
                    if isinstance(event['timestamp'], str):
                        event['timestamp'] = datetime.fromisoformat(event['timestamp'])
                    
                    # Validate event structure
                    validate_player_event(event)
                    validated_events.append(event)
                    
                except Exception as e:
                    validation_errors.append(f"Event {i}: {e}")
            
            # Check validation error threshold
            if validation_errors:
                error_rate = len(validation_errors) / len(events_data)
                if error_rate > 0.05:  # More than 5% errors
                    raise ValueError(f"Too many event validation errors ({error_rate:.1%}): {validation_errors[:5]}")
                else:
                    logger.warning(f"Event validation errors ({error_rate:.1%}): {validation_errors[:3]}")
            
            # Run data quality validation
            events_df = pd.DataFrame(validated_events)
            quality_results = self.validator.validate_events_data(events_df)
            
            # Check for critical quality issues
            critical_issues = [r for r in quality_results if not r.passed and r.severity == ValidationSeverity.CRITICAL]
            if critical_issues:
                raise ValueError(f"Critical data quality issues: {[r.message for r in critical_issues]}")
            
            logger.info(f"Loaded {len(validated_events)} player events with {len(validation_errors)} validation errors")
            return validated_events
        
        return self.job_wrapper.execute_job(
            job_id=job_id,
            job_type="load_player_events",
            job_func=_load_events_job,
            input_data=self.data_dir,
            timeout_seconds=120
        )
    
    def load_churn_features(self) -> List[ChurnFeatures]:
        """Load and validate churn features from JSON."""
        features_file = self.data_dir / "churn_features.json"
        
        if not features_file.exists():
            raise FileNotFoundError(f"Churn features file not found: {features_file}")
        
        with open(features_file, 'r') as f:
            features_data = json.load(f)
        
        features = []
        for feature_data in features_data:
            # Convert date string to date object
            if isinstance(feature_data['feature_date'], str):
                feature_data['feature_date'] = date.fromisoformat(feature_data['feature_date'])
            
            feature = ChurnFeatures.from_dict(feature_data)
            features.append(feature)
        
        return features
    
    def load_all_data(self) -> Tuple[List[PlayerProfile], List[Dict[str, Any]], List[ChurnFeatures]]:
        """Load all data types and return as tuple."""
        profiles = self.load_player_profiles()
        events = self.load_player_events()
        features = self.load_churn_features()
        
        return profiles, events, features


class EventIngestion:
    """Processes and ingests raw player events for analytics."""
    
    def __init__(self):
        """Initialize event ingestion processor."""
        pass
    
    def process_events_to_dataframe(self, events: List[Dict[str, Any]]) -> pd.DataFrame:
        """Convert events list to pandas DataFrame for analysis."""
        df = pd.DataFrame(events)
        
        # Ensure timestamp is datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Add derived columns
        df['date'] = df['timestamp'].dt.date
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        
        return df
    
    def extract_session_events(self, events_df: pd.DataFrame) -> pd.DataFrame:
        """Extract and process session start/end events."""
        session_events = events_df[
            events_df['event_type'].isin(['session_start', 'session_end'])
        ].copy()
        
        # Sort by player and timestamp
        session_events = session_events.sort_values(['player_id', 'timestamp'])
        
        return session_events
    
    def extract_purchase_events(self, events_df: pd.DataFrame) -> pd.DataFrame:
        """Extract and process purchase events."""
        purchase_events = events_df[
            events_df['event_type'] == 'purchase'
        ].copy()
        
        return purchase_events
    
    def extract_level_events(self, events_df: pd.DataFrame) -> pd.DataFrame:
        """Extract and process level completion events."""
        level_events = events_df[
            events_df['event_type'] == 'level_complete'
        ].copy()
        
        return level_events
    
    def calculate_session_metrics(self, session_events_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate session-level metrics from session events."""
        # Group session starts and ends
        session_starts = session_events_df[
            session_events_df['event_type'] == 'session_start'
        ].copy()
        
        # Calculate daily session metrics per player
        daily_sessions = session_starts.groupby(['player_id', 'date']).agg({
            'timestamp': 'count',  # Number of sessions
            'session_duration': ['mean', 'sum']  # Avg and total duration
        }).reset_index()
        
        # Flatten column names
        daily_sessions.columns = [
            'player_id', 'date', 'session_count', 
            'avg_session_duration', 'total_session_duration'
        ]
        
        return daily_sessions
    
    def calculate_purchase_metrics(self, purchase_events_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate purchase metrics from purchase events."""
        if purchase_events_df.empty:
            return pd.DataFrame(columns=['player_id', 'date', 'purchase_count', 'total_revenue'])
        
        daily_purchases = purchase_events_df.groupby(['player_id', 'date']).agg({
            'purchase_amount': ['count', 'sum']
        }).reset_index()
        
        # Flatten column names
        daily_purchases.columns = ['player_id', 'date', 'purchase_count', 'total_revenue']
        
        return daily_purchases
    
    def calculate_progression_metrics(self, level_events_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate level progression metrics."""
        if level_events_df.empty:
            return pd.DataFrame(columns=['player_id', 'date', 'levels_completed', 'max_level'])
        
        daily_progression = level_events_df.groupby(['player_id', 'date']).agg({
            'level': ['count', 'max']
        }).reset_index()
        
        # Flatten column names
        daily_progression.columns = ['player_id', 'date', 'levels_completed', 'max_level']
        
        return daily_progression
    
    def create_daily_player_summary(self, events_df: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive daily summary for each player."""
        
        # Extract different event types
        session_events = self.extract_session_events(events_df)
        purchase_events = self.extract_purchase_events(events_df)
        level_events = self.extract_level_events(events_df)
        
        # Calculate metrics for each event type
        session_metrics = self.calculate_session_metrics(session_events)
        purchase_metrics = self.calculate_purchase_metrics(purchase_events)
        progression_metrics = self.calculate_progression_metrics(level_events)
        
        # Get all unique player-date combinations
        all_dates = events_df.groupby(['player_id', 'date']).size().reset_index(name='total_events')
        
        # Merge all metrics
        daily_summary = all_dates
        
        # Merge session metrics
        daily_summary = daily_summary.merge(
            session_metrics, on=['player_id', 'date'], how='left'
        )
        
        # Merge purchase metrics
        daily_summary = daily_summary.merge(
            purchase_metrics, on=['player_id', 'date'], how='left'
        )
        
        # Merge progression metrics
        daily_summary = daily_summary.merge(
            progression_metrics, on=['player_id', 'date'], how='left'
        )
        
        # Fill NaN values with 0
        numeric_columns = [
            'session_count', 'avg_session_duration', 'total_session_duration',
            'purchase_count', 'total_revenue', 'levels_completed', 'max_level'
        ]
        
        for col in numeric_columns:
            if col in daily_summary.columns:
                daily_summary[col] = daily_summary[col].fillna(0)
                daily_summary[col] = daily_summary[col].astype('float64')
        
        return daily_summary
    
    def validate_data_quality(self, events_df: pd.DataFrame, 
                            profiles: List[PlayerProfile]) -> Dict[str, Any]:
        """Validate data quality and return quality metrics."""
        
        quality_report = {
            'total_events': len(events_df),
            'total_players': len(profiles),
            'date_range': {
                'start': events_df['timestamp'].min(),
                'end': events_df['timestamp'].max()
            },
            'event_types': events_df['event_type'].value_counts().to_dict(),
            'issues': []
        }
        
        # Check for missing player IDs
        event_players = set(events_df['player_id'].unique())
        profile_players = set(p.player_id for p in profiles)
        
        missing_profiles = event_players - profile_players
        missing_events = profile_players - event_players
        
        if missing_profiles:
            quality_report['issues'].append(
                f"Events found for {len(missing_profiles)} players without profiles"
            )
        
        if missing_events:
            quality_report['issues'].append(
                f"{len(missing_events)} players have profiles but no events"
            )
        
        # Check for duplicate events
        duplicate_events = events_df.duplicated().sum()
        if duplicate_events > 0:
            quality_report['issues'].append(f"{duplicate_events} duplicate events found")
        
        # Check for future timestamps
        future_events = events_df[events_df['timestamp'] > datetime.now()]
        if len(future_events) > 0:
            quality_report['issues'].append(f"{len(future_events)} events have future timestamps")
        
        return quality_report