"""
Data validation with configurable quality thresholds.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationRule:
    """Configuration for a data validation rule."""
    name: str
    description: str
    threshold: float
    severity: ValidationSeverity
    enabled: bool = True


@dataclass
class ValidationResult:
    """Result of a data validation check."""
    rule_name: str
    passed: bool
    value: float
    threshold: float
    severity: ValidationSeverity
    message: str
    details: Optional[Dict[str, Any]] = None


class DataQualityConfig:
    """Configuration for data quality validation thresholds."""
    
    def __init__(self):
        self.rules = {
            # Completeness rules
            'missing_player_ids': ValidationRule(
                name='missing_player_ids',
                description='Percentage of events with missing player IDs',
                threshold=0.01,  # Max 1% missing
                severity=ValidationSeverity.ERROR
            ),
            'missing_timestamps': ValidationRule(
                name='missing_timestamps',
                description='Percentage of events with missing timestamps',
                threshold=0.001,  # Max 0.1% missing
                severity=ValidationSeverity.CRITICAL
            ),
            'missing_event_types': ValidationRule(
                name='missing_event_types',
                description='Percentage of events with missing event types',
                threshold=0.001,  # Max 0.1% missing
                severity=ValidationSeverity.CRITICAL
            ),
            
            # Consistency rules
            'duplicate_events': ValidationRule(
                name='duplicate_events',
                description='Percentage of duplicate events',
                threshold=0.05,  # Max 5% duplicates
                severity=ValidationSeverity.WARNING
            ),
            'future_timestamps': ValidationRule(
                name='future_timestamps',
                description='Percentage of events with future timestamps',
                threshold=0.001,  # Max 0.1% future events
                severity=ValidationSeverity.ERROR
            ),
            'invalid_session_durations': ValidationRule(
                name='invalid_session_durations',
                description='Percentage of sessions with invalid durations',
                threshold=0.02,  # Max 2% invalid durations
                severity=ValidationSeverity.WARNING
            ),
            
            # Business logic rules
            'orphaned_events': ValidationRule(
                name='orphaned_events',
                description='Percentage of events without corresponding player profiles',
                threshold=0.05,  # Max 5% orphaned events
                severity=ValidationSeverity.WARNING
            ),
            'inactive_player_events': ValidationRule(
                name='inactive_player_events',
                description='Percentage of events from players inactive >90 days',
                threshold=0.1,  # Max 10% from inactive players
                severity=ValidationSeverity.INFO
            ),
            
            # Volume rules
            'daily_event_volume_drop': ValidationRule(
                name='daily_event_volume_drop',
                description='Daily event volume drop percentage',
                threshold=0.3,  # Max 30% drop
                severity=ValidationSeverity.ERROR
            ),
            'hourly_event_volume_spike': ValidationRule(
                name='hourly_event_volume_spike',
                description='Hourly event volume spike percentage',
                threshold=5.0,  # Max 500% spike
                severity=ValidationSeverity.WARNING
            )
        }
    
    def get_rule(self, rule_name: str) -> Optional[ValidationRule]:
        """Get validation rule by name."""
        return self.rules.get(rule_name)
    
    def update_rule_threshold(self, rule_name: str, threshold: float) -> bool:
        """Update threshold for a validation rule."""
        if rule_name in self.rules:
            self.rules[rule_name].threshold = threshold
            return True
        return False
    
    def enable_rule(self, rule_name: str, enabled: bool = True) -> bool:
        """Enable or disable a validation rule."""
        if rule_name in self.rules:
            self.rules[rule_name].enabled = enabled
            return True
        return False


class DataQualityValidator:
    """Validates data quality against configurable thresholds."""
    
    def __init__(self, config: Optional[DataQualityConfig] = None):
        self.config = config or DataQualityConfig()
        self.validation_history: List[Dict[str, Any]] = []
    
    def validate_events_data(self, events_df: pd.DataFrame) -> List[ValidationResult]:
        """Validate events data quality."""
        results = []
        
        if events_df.empty:
            results.append(ValidationResult(
                rule_name='empty_dataset',
                passed=False,
                value=0,
                threshold=1,
                severity=ValidationSeverity.CRITICAL,
                message='Events dataset is empty'
            ))
            return results
        
        # Check missing player IDs
        results.append(self._check_missing_values(
            events_df, 'player_id', 'missing_player_ids'
        ))
        
        # Check missing timestamps
        results.append(self._check_missing_values(
            events_df, 'timestamp', 'missing_timestamps'
        ))
        
        # Check missing event types
        results.append(self._check_missing_values(
            events_df, 'event_type', 'missing_event_types'
        ))
        
        # Check for duplicates
        results.append(self._check_duplicates(events_df))
        
        # Check for future timestamps
        results.append(self._check_future_timestamps(events_df))
        
        # Check session durations
        results.append(self._check_session_durations(events_df))
        
        return [r for r in results if r is not None]
    
    def validate_profiles_data(self, profiles_df: pd.DataFrame) -> List[ValidationResult]:
        """Validate player profiles data quality."""
        results = []
        
        if profiles_df.empty:
            results.append(ValidationResult(
                rule_name='empty_profiles',
                passed=False,
                value=0,
                threshold=1,
                severity=ValidationSeverity.ERROR,
                message='Player profiles dataset is empty'
            ))
            return results
        
        # Check for duplicate player IDs
        duplicate_count = profiles_df['player_id'].duplicated().sum()
        duplicate_percentage = duplicate_count / len(profiles_df)
        
        results.append(ValidationResult(
            rule_name='duplicate_player_profiles',
            passed=duplicate_percentage == 0,
            value=duplicate_percentage,
            threshold=0,
            severity=ValidationSeverity.ERROR,
            message=f'Found {duplicate_count} duplicate player profiles ({duplicate_percentage:.2%})'
        ))
        
        return results
    
    def validate_data_consistency(self, events_df: pd.DataFrame, 
                                profiles_df: pd.DataFrame) -> List[ValidationResult]:
        """Validate consistency between events and profiles data."""
        results = []
        
        # Check for orphaned events
        event_players = set(events_df['player_id'].unique())
        profile_players = set(profiles_df['player_id'].unique())
        
        orphaned_players = event_players - profile_players
        orphaned_events = events_df[events_df['player_id'].isin(orphaned_players)]
        orphaned_percentage = len(orphaned_events) / len(events_df)
        
        rule = self.config.get_rule('orphaned_events')
        if rule and rule.enabled:
            results.append(ValidationResult(
                rule_name='orphaned_events',
                passed=orphaned_percentage <= rule.threshold,
                value=orphaned_percentage,
                threshold=rule.threshold,
                severity=rule.severity,
                message=f'Found {len(orphaned_events)} orphaned events ({orphaned_percentage:.2%})',
                details={'orphaned_player_count': len(orphaned_players)}
            ))
        
        return results
    
    def validate_volume_patterns(self, events_df: pd.DataFrame, 
                               historical_data: Optional[pd.DataFrame] = None) -> List[ValidationResult]:
        """Validate event volume patterns for anomalies."""
        results = []
        
        # Daily volume check
        if historical_data is not None:
            results.extend(self._check_daily_volume_patterns(events_df, historical_data))
        
        # Hourly volume spikes
        results.append(self._check_hourly_volume_spikes(events_df))
        
        return [r for r in results if r is not None]
    
    def _check_missing_values(self, df: pd.DataFrame, column: str, 
                            rule_name: str) -> Optional[ValidationResult]:
        """Check for missing values in a column."""
        rule = self.config.get_rule(rule_name)
        if not rule or not rule.enabled:
            return None
        
        missing_count = df[column].isnull().sum()
        missing_percentage = missing_count / len(df)
        
        return ValidationResult(
            rule_name=rule_name,
            passed=missing_percentage <= rule.threshold,
            value=missing_percentage,
            threshold=rule.threshold,
            severity=rule.severity,
            message=f'Found {missing_count} missing {column} values ({missing_percentage:.2%})'
        )
    
    def _check_duplicates(self, df: pd.DataFrame) -> Optional[ValidationResult]:
        """Check for duplicate events."""
        rule = self.config.get_rule('duplicate_events')
        if not rule or not rule.enabled:
            return None
        
        duplicate_count = df.duplicated().sum()
        duplicate_percentage = duplicate_count / len(df)
        
        return ValidationResult(
            rule_name='duplicate_events',
            passed=duplicate_percentage <= rule.threshold,
            value=duplicate_percentage,
            threshold=rule.threshold,
            severity=rule.severity,
            message=f'Found {duplicate_count} duplicate events ({duplicate_percentage:.2%})'
        )
    
    def _check_future_timestamps(self, df: pd.DataFrame) -> Optional[ValidationResult]:
        """Check for events with future timestamps."""
        rule = self.config.get_rule('future_timestamps')
        if not rule or not rule.enabled:
            return None
        
        now = datetime.now()
        future_events = df[pd.to_datetime(df['timestamp']) > now]
        future_percentage = len(future_events) / len(df)
        
        return ValidationResult(
            rule_name='future_timestamps',
            passed=future_percentage <= rule.threshold,
            value=future_percentage,
            threshold=rule.threshold,
            severity=rule.severity,
            message=f'Found {len(future_events)} events with future timestamps ({future_percentage:.2%})'
        )
    
    def _check_session_durations(self, df: pd.DataFrame) -> Optional[ValidationResult]:
        """Check for invalid session durations."""
        rule = self.config.get_rule('invalid_session_durations')
        if not rule or not rule.enabled:
            return None
        
        session_events = df[df['event_type'].isin(['session_start', 'session_end'])]
        if session_events.empty or 'session_duration' not in session_events.columns:
            return ValidationResult(
                rule_name='invalid_session_durations',
                passed=True,
                value=0.0,
                threshold=rule.threshold,
                severity=rule.severity,
                message='No session duration data available for validation'
            )
        
        # Check for negative or extremely long durations
        invalid_durations = session_events[
            (session_events['session_duration'] < 0) | 
            (session_events['session_duration'] > 86400)  # More than 24 hours
        ]
        
        invalid_percentage = len(invalid_durations) / len(session_events)
        
        return ValidationResult(
            rule_name='invalid_session_durations',
            passed=invalid_percentage <= rule.threshold,
            value=invalid_percentage,
            threshold=rule.threshold,
            severity=rule.severity,
            message=f'Found {len(invalid_durations)} sessions with invalid durations ({invalid_percentage:.2%})'
        )
    
    def _check_daily_volume_patterns(self, current_df: pd.DataFrame, 
                                   historical_df: pd.DataFrame) -> List[ValidationResult]:
        """Check for unusual daily volume patterns."""
        results = []
        rule = self.config.get_rule('daily_event_volume_drop')
        if not rule or not rule.enabled:
            return results
        
        # Calculate daily volumes
        current_df['date'] = pd.to_datetime(current_df['timestamp']).dt.date
        historical_df['date'] = pd.to_datetime(historical_df['timestamp']).dt.date
        
        current_daily = current_df.groupby('date').size()
        historical_daily = historical_df.groupby('date').size()
        
        if len(historical_daily) > 0:
            historical_avg = historical_daily.mean()
            
            for date, volume in current_daily.items():
                volume_drop = (historical_avg - volume) / historical_avg
                
                if volume_drop > rule.threshold:
                    results.append(ValidationResult(
                        rule_name='daily_event_volume_drop',
                        passed=False,
                        value=volume_drop,
                        threshold=rule.threshold,
                        severity=rule.severity,
                        message=f'Daily volume drop of {volume_drop:.1%} on {date}',
                        details={'date': str(date), 'volume': volume, 'historical_avg': historical_avg}
                    ))
        
        return results
    
    def _check_hourly_volume_spikes(self, df: pd.DataFrame) -> Optional[ValidationResult]:
        """Check for unusual hourly volume spikes."""
        rule = self.config.get_rule('hourly_event_volume_spike')
        if not rule or not rule.enabled:
            return None
        
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        hourly_volumes = df.groupby('hour').size()
        
        if len(hourly_volumes) > 1:
            avg_volume = hourly_volumes.mean()
            max_volume = hourly_volumes.max()
            spike_ratio = max_volume / avg_volume if avg_volume > 0 else 0
            
            return ValidationResult(
                rule_name='hourly_event_volume_spike',
                passed=spike_ratio <= rule.threshold,
                value=spike_ratio,
                threshold=rule.threshold,
                severity=rule.severity,
                message=f'Hourly volume spike of {spike_ratio:.1f}x average detected'
            )
        
        return None
    
    def generate_quality_report(self, validation_results: List[ValidationResult]) -> Dict[str, Any]:
        """Generate comprehensive data quality report."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_checks': len(validation_results),
            'passed_checks': sum(1 for r in validation_results if r.passed),
            'failed_checks': sum(1 for r in validation_results if not r.passed),
            'severity_breakdown': {},
            'issues': [],
            'recommendations': []
        }
        
        # Count by severity
        for severity in ValidationSeverity:
            count = sum(1 for r in validation_results 
                       if r.severity == severity and not r.passed)
            report['severity_breakdown'][severity.value] = count
        
        # Collect failed checks
        for result in validation_results:
            if not result.passed:
                report['issues'].append({
                    'rule': result.rule_name,
                    'severity': result.severity.value,
                    'message': result.message,
                    'value': result.value,
                    'threshold': result.threshold,
                    'details': result.details
                })
        
        # Generate recommendations
        report['recommendations'] = self._generate_recommendations(validation_results)
        
        # Store in history
        self.validation_history.append(report)
        
        return report
    
    def _generate_recommendations(self, validation_results: List[ValidationResult]) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        critical_issues = [r for r in validation_results 
                          if not r.passed and r.severity == ValidationSeverity.CRITICAL]
        error_issues = [r for r in validation_results 
                       if not r.passed and r.severity == ValidationSeverity.ERROR]
        
        if critical_issues:
            recommendations.append("CRITICAL: Address data completeness issues immediately")
        
        if error_issues:
            recommendations.append("ERROR: Review data ingestion pipeline for consistency issues")
        
        # Specific recommendations
        for result in validation_results:
            if not result.passed:
                if result.rule_name == 'orphaned_events':
                    recommendations.append("Ensure player profile creation precedes event ingestion")
                elif result.rule_name == 'duplicate_events':
                    recommendations.append("Implement event deduplication in ingestion pipeline")
                elif result.rule_name == 'future_timestamps':
                    recommendations.append("Validate timestamp formats and timezone handling")
        
        return recommendations