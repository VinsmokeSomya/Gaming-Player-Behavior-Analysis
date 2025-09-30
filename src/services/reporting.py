"""Automated reporting system for player retention analytics.

This module provides automated daily reporting capabilities including:
- Daily retention summary reports
- Key metrics calculation (DAU, WAU, MAU, churn rate)
- Historical baseline comparison
- Automated alerting for unusual retention patterns
"""

import logging
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from src.analytics.query_engine import query_engine, RetentionQueryResult
from src.database import db_manager
from src.config import app_config

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class KeyMetrics:
    """Container for key retention metrics."""
    date: date
    dau: int
    wau: int
    mau: int
    churn_rate: float
    day_1_retention: float
    day_7_retention: float
    day_30_retention: float
    new_registrations: int
    total_active_players: int


@dataclass
class BaselineComparison:
    """Container for baseline comparison results."""
    metric_name: str
    current_value: float
    baseline_value: float
    percentage_change: float
    is_significant: bool
    threshold_breached: bool


@dataclass
class RetentionAlert:
    """Container for retention alerts."""
    alert_id: str
    severity: AlertSeverity
    metric_name: str
    current_value: float
    expected_value: float
    deviation_percentage: float
    message: str
    timestamp: datetime
    resolved: bool = False


@dataclass
class DailyRetentionReport:
    """Container for daily retention summary report."""
    report_date: date
    key_metrics: KeyMetrics
    baseline_comparisons: List[BaselineComparison]
    alerts: List[RetentionAlert]
    cohort_analysis: List[RetentionQueryResult]
    summary: str
    generated_at: datetime


class MetricsCalculator:
    """Calculator for key retention metrics."""
    
    def __init__(self):
        self.query_engine = query_engine
        self.db_manager = db_manager
    
    def calculate_dau(self, target_date: date) -> int:
        """Calculate Daily Active Users for a specific date."""
        try:
            dau_data = self.query_engine.get_daily_active_users(target_date, target_date)
            return dau_data[0][1] if dau_data else 0
        except Exception as e:
            logger.error(f"Error calculating DAU for {target_date}: {e}")
            return 0
    
    def calculate_wau(self, target_date: date) -> int:
        """Calculate Weekly Active Users for the week containing target_date."""
        try:
            # Get the start of the week containing target_date
            week_start = target_date - timedelta(days=target_date.weekday())
            week_end = week_start + timedelta(days=6)
            
            wau_data = self.query_engine.get_weekly_active_users(week_start, week_end)
            return wau_data[0][1] if wau_data else 0
        except Exception as e:
            logger.error(f"Error calculating WAU for {target_date}: {e}")
            return 0
    
    def calculate_mau(self, target_date: date) -> int:
        """Calculate Monthly Active Users for the month containing target_date."""
        try:
            # Get the start and end of the month containing target_date
            month_start = target_date.replace(day=1)
            if target_date.month == 12:
                month_end = target_date.replace(year=target_date.year + 1, month=1, day=1) - timedelta(days=1)
            else:
                month_end = target_date.replace(month=target_date.month + 1, day=1) - timedelta(days=1)
            
            query = """
            SELECT COUNT(DISTINCT ps.player_id) as mau_count
            FROM player_sessions ps
            WHERE DATE(ps.start_time) BETWEEN :start_date AND :end_date
            """
            
            params = {
                'start_date': month_start,
                'end_date': min(month_end, target_date)  # Don't count future dates
            }
            
            with self.db_manager.get_session() as session:
                from sqlalchemy import text
                result = session.execute(text(query), params)
                row = result.fetchone()
                return row.mau_count if row else 0
                
        except Exception as e:
            logger.error(f"Error calculating MAU for {target_date}: {e}")
            return 0
    
    def calculate_churn_rate(self, target_date: date, lookback_days: int = 7) -> float:
        """Calculate churn rate based on players inactive for lookback_days."""
        try:
            cutoff_date = target_date - timedelta(days=lookback_days)
            
            query = """
            WITH active_players AS (
                SELECT DISTINCT player_id
                FROM player_sessions
                WHERE DATE(start_time) >= :cutoff_date
            ),
            total_players AS (
                SELECT COUNT(*) as total_count
                FROM player_profiles
                WHERE registration_date <= :target_date
            ),
            churned_players AS (
                SELECT COUNT(*) as churned_count
                FROM player_profiles pp
                WHERE pp.registration_date <= :target_date
                AND pp.player_id NOT IN (SELECT player_id FROM active_players)
                AND pp.last_active_date < :cutoff_date
            )
            SELECT 
                CASE 
                    WHEN tp.total_count > 0 THEN cp.churned_count::DECIMAL / tp.total_count
                    ELSE 0 
                END as churn_rate
            FROM total_players tp, churned_players cp
            """
            
            params = {
                'target_date': target_date,
                'cutoff_date': cutoff_date
            }
            
            with self.db_manager.get_session() as session:
                from sqlalchemy import text
                result = session.execute(text(query), params)
                row = result.fetchone()
                return float(row.churn_rate) if row else 0.0
                
        except Exception as e:
            logger.error(f"Error calculating churn rate for {target_date}: {e}")
            return 0.0
    
    def calculate_new_registrations(self, target_date: date) -> int:
        """Calculate new player registrations for a specific date."""
        try:
            query = """
            SELECT COUNT(*) as new_registrations
            FROM player_profiles
            WHERE DATE(registration_date) = :target_date
            """
            
            params = {'target_date': target_date}
            
            with self.db_manager.get_session() as session:
                from sqlalchemy import text
                result = session.execute(text(query), params)
                row = result.fetchone()
                return row.new_registrations if row else 0
                
        except Exception as e:
            logger.error(f"Error calculating new registrations for {target_date}: {e}")
            return 0
    
    def calculate_retention_rates(self, target_date: date) -> Tuple[float, float, float]:
        """Calculate average retention rates for cohorts ending on target_date."""
        try:
            # Calculate retention for cohorts that would have their retention measured by target_date
            cohort_1_day = target_date - timedelta(days=1)
            cohort_7_day = target_date - timedelta(days=7)
            cohort_30_day = target_date - timedelta(days=30)
            
            # Get retention data for recent cohorts
            retention_data = self.query_engine.calculate_cohort_retention(
                start_date=cohort_30_day,
                end_date=cohort_1_day
            )
            
            if not retention_data:
                return 0.0, 0.0, 0.0
            
            # Calculate weighted averages based on cohort sizes
            total_cohort_size = sum(r.cohort_size for r in retention_data)
            if total_cohort_size == 0:
                return 0.0, 0.0, 0.0
            
            day_1_retention = sum(r.day_1_retention * r.cohort_size for r in retention_data) / total_cohort_size
            day_7_retention = sum(r.day_7_retention * r.cohort_size for r in retention_data) / total_cohort_size
            day_30_retention = sum(r.day_30_retention * r.cohort_size for r in retention_data) / total_cohort_size
            
            return day_1_retention, day_7_retention, day_30_retention
            
        except Exception as e:
            logger.error(f"Error calculating retention rates for {target_date}: {e}")
            return 0.0, 0.0, 0.0


class BaselineManager:
    """Manages historical baselines for metrics comparison."""
    
    def __init__(self):
        self.db_manager = db_manager
    
    def calculate_baseline(self, metric_name: str, target_date: date, lookback_days: int = 30) -> float:
        """Calculate baseline value for a metric using historical data."""
        try:
            start_date = target_date - timedelta(days=lookback_days)
            end_date = target_date - timedelta(days=1)  # Exclude current date
            
            query = """
            SELECT AVG(metric_value) as baseline_value
            FROM daily_metrics_history
            WHERE metric_name = :metric_name
            AND metric_date BETWEEN :start_date AND :end_date
            """
            
            params = {
                'metric_name': metric_name,
                'start_date': start_date,
                'end_date': end_date
            }
            
            with self.db_manager.get_session() as session:
                from sqlalchemy import text
                result = session.execute(text(query), params)
                row = result.fetchone()
                return float(row.baseline_value) if row and row.baseline_value else 0.0
                
        except Exception as e:
            logger.warning(f"Error calculating baseline for {metric_name}: {e}")
            # Fallback: return the current value as baseline if no history available
            return 0.0
    
    def store_daily_metrics(self, metrics: KeyMetrics) -> None:
        """Store daily metrics for future baseline calculations."""
        try:
            metrics_to_store = [
                ('dau', metrics.dau),
                ('wau', metrics.wau),
                ('mau', metrics.mau),
                ('churn_rate', metrics.churn_rate),
                ('day_1_retention', metrics.day_1_retention),
                ('day_7_retention', metrics.day_7_retention),
                ('day_30_retention', metrics.day_30_retention),
                ('new_registrations', metrics.new_registrations),
                ('total_active_players', metrics.total_active_players)
            ]
            
            query = """
            INSERT INTO daily_metrics_history (metric_date, metric_name, metric_value, created_at)
            VALUES (:metric_date, :metric_name, :metric_value, :created_at)
            ON CONFLICT (metric_date, metric_name) 
            DO UPDATE SET metric_value = EXCLUDED.metric_value, updated_at = :created_at
            """
            
            with self.db_manager.get_session() as session:
                from sqlalchemy import text
                for metric_name, metric_value in metrics_to_store:
                    params = {
                        'metric_date': metrics.date,
                        'metric_name': metric_name,
                        'metric_value': float(metric_value),
                        'created_at': datetime.now()
                    }
                    session.execute(text(query), params)
                session.commit()
                
        except Exception as e:
            logger.error(f"Error storing daily metrics: {e}")
    
    def compare_with_baseline(self, metric_name: str, current_value: float, target_date: date) -> BaselineComparison:
        """Compare current metric value with historical baseline."""
        baseline_value = self.calculate_baseline(metric_name, target_date)
        
        if baseline_value == 0:
            percentage_change = 0.0
            is_significant = False
        else:
            percentage_change = ((current_value - baseline_value) / baseline_value) * 100
            is_significant = abs(percentage_change) >= 10.0  # 10% threshold for significance
        
        # Define thresholds for different metrics
        threshold_breached = False
        if metric_name in ['dau', 'wau', 'mau', 'day_1_retention', 'day_7_retention', 'day_30_retention']:
            threshold_breached = percentage_change < -20.0  # 20% decrease is concerning
        elif metric_name == 'churn_rate':
            threshold_breached = percentage_change > 25.0  # 25% increase in churn is concerning
        
        return BaselineComparison(
            metric_name=metric_name,
            current_value=current_value,
            baseline_value=baseline_value,
            percentage_change=percentage_change,
            is_significant=is_significant,
            threshold_breached=threshold_breached
        )


class AlertManager:
    """Manages automated alerts for unusual retention patterns."""
    
    def __init__(self):
        self.db_manager = db_manager
    
    def generate_alerts(self, metrics: KeyMetrics, comparisons: List[BaselineComparison]) -> List[RetentionAlert]:
        """Generate alerts based on metrics and baseline comparisons."""
        alerts = []
        
        for comparison in comparisons:
            if comparison.threshold_breached:
                severity = self._determine_severity(comparison)
                alert = RetentionAlert(
                    alert_id=f"{comparison.metric_name}_{metrics.date}_{datetime.now().strftime('%H%M%S%f')}",
                    severity=severity,
                    metric_name=comparison.metric_name,
                    current_value=comparison.current_value,
                    expected_value=comparison.baseline_value,
                    deviation_percentage=comparison.percentage_change,
                    message=self._generate_alert_message(comparison),
                    timestamp=datetime.now()
                )
                alerts.append(alert)
        
        # Additional custom alert logic
        if metrics.churn_rate > 0.3:  # 30% churn rate is critical
            alerts.append(RetentionAlert(
                alert_id=f"high_churn_{metrics.date}_{datetime.now().strftime('%H%M%S%f')}",
                severity=AlertSeverity.CRITICAL,
                metric_name="churn_rate",
                current_value=metrics.churn_rate,
                expected_value=0.15,  # Expected churn rate
                deviation_percentage=((metrics.churn_rate - 0.15) / 0.15) * 100,
                message=f"Critical: Churn rate has reached {metrics.churn_rate:.1%}, immediate action required",
                timestamp=datetime.now()
            ))
        
        if metrics.dau < 100:  # Very low DAU
            alerts.append(RetentionAlert(
                alert_id=f"low_dau_{metrics.date}_{datetime.now().strftime('%H%M%S%f')}",
                severity=AlertSeverity.HIGH,
                metric_name="dau",
                current_value=metrics.dau,
                expected_value=500,  # Expected minimum DAU
                deviation_percentage=((metrics.dau - 500) / 500) * 100,
                message=f"High: Daily active users dropped to {metrics.dau}, well below expected levels",
                timestamp=datetime.now()
            ))
        
        return alerts
    
    def _determine_severity(self, comparison: BaselineComparison) -> AlertSeverity:
        """Determine alert severity based on comparison results."""
        abs_change = abs(comparison.percentage_change)
        
        if abs_change >= 50:
            return AlertSeverity.CRITICAL
        elif abs_change >= 30:
            return AlertSeverity.HIGH
        elif abs_change >= 20:
            return AlertSeverity.MEDIUM
        else:
            return AlertSeverity.LOW
    
    def _generate_alert_message(self, comparison: BaselineComparison) -> str:
        """Generate human-readable alert message."""
        direction = "increased" if comparison.percentage_change > 0 else "decreased"
        return (f"{comparison.metric_name.replace('_', ' ').title()} has {direction} by "
                f"{abs(comparison.percentage_change):.1f}% from baseline "
                f"({comparison.baseline_value:.2f} to {comparison.current_value:.2f})")
    
    def store_alerts(self, alerts: List[RetentionAlert]) -> None:
        """Store alerts in database for tracking and resolution."""
        if not alerts:
            return
        
        try:
            query = """
            INSERT INTO retention_alerts (
                alert_id, severity, metric_name, current_value, expected_value,
                deviation_percentage, message, timestamp, resolved, created_at
            ) VALUES (
                :alert_id, :severity, :metric_name, :current_value, :expected_value,
                :deviation_percentage, :message, :timestamp, :resolved, :created_at
            )
            """
            
            with self.db_manager.get_session() as session:
                from sqlalchemy import text
                for alert in alerts:
                    params = {
                        'alert_id': alert.alert_id,
                        'severity': alert.severity.value,
                        'metric_name': alert.metric_name,
                        'current_value': alert.current_value,
                        'expected_value': alert.expected_value,
                        'deviation_percentage': alert.deviation_percentage,
                        'message': alert.message,
                        'timestamp': alert.timestamp,
                        'resolved': alert.resolved,
                        'created_at': datetime.now()
                    }
                    session.execute(text(query), params)
                session.commit()
                
        except Exception as e:
            logger.error(f"Error storing alerts: {e}")
    
    def send_alert_notifications(self, alerts: List[RetentionAlert]) -> None:
        """Send alert notifications via email or other channels."""
        if not alerts:
            return
        
        # Filter alerts by severity for notification
        critical_alerts = [a for a in alerts if a.severity == AlertSeverity.CRITICAL]
        high_alerts = [a for a in alerts if a.severity == AlertSeverity.HIGH]
        
        if critical_alerts or high_alerts:
            try:
                self._send_email_alerts(critical_alerts + high_alerts)
            except Exception as e:
                logger.error(f"Error sending alert notifications: {e}")
    
    def _send_email_alerts(self, alerts: List[RetentionAlert]) -> None:
        """Send email notifications for alerts."""
        # This is a placeholder implementation
        # In production, you would configure SMTP settings and recipient lists
        logger.info(f"Would send email alerts for {len(alerts)} alerts:")
        for alert in alerts:
            logger.info(f"  - {alert.severity.value.upper()}: {alert.message}")


class ReportGenerator:
    """Generates automated daily retention reports."""
    
    def __init__(self):
        self.metrics_calculator = MetricsCalculator()
        self.baseline_manager = BaselineManager()
        self.alert_manager = AlertManager()
        self.query_engine = query_engine
        self.db_manager = db_manager
    
    def generate_daily_report(self, target_date: Optional[date] = None) -> DailyRetentionReport:
        """Generate comprehensive daily retention report."""
        if target_date is None:
            target_date = date.today() - timedelta(days=1)  # Previous day by default
        
        logger.info(f"Generating daily retention report for {target_date}")
        
        try:
            # Calculate key metrics
            key_metrics = self._calculate_key_metrics(target_date)
            
            # Store metrics for future baseline calculations
            self.baseline_manager.store_daily_metrics(key_metrics)
            
            # Compare with baselines
            baseline_comparisons = self._generate_baseline_comparisons(key_metrics, target_date)
            
            # Generate alerts
            alerts = self.alert_manager.generate_alerts(key_metrics, baseline_comparisons)
            
            # Store and send alerts
            self.alert_manager.store_alerts(alerts)
            self.alert_manager.send_alert_notifications(alerts)
            
            # Get cohort analysis
            cohort_analysis = self._get_cohort_analysis(target_date)
            
            # Generate summary
            summary = self._generate_summary(key_metrics, baseline_comparisons, alerts)
            
            report = DailyRetentionReport(
                report_date=target_date,
                key_metrics=key_metrics,
                baseline_comparisons=baseline_comparisons,
                alerts=alerts,
                cohort_analysis=cohort_analysis,
                summary=summary,
                generated_at=datetime.now()
            )
            
            # Store report
            self._store_report(report)
            
            logger.info(f"Daily retention report generated successfully for {target_date}")
            return report
            
        except Exception as e:
            logger.error(f"Error generating daily report for {target_date}: {e}")
            raise
    
    def _calculate_key_metrics(self, target_date: date) -> KeyMetrics:
        """Calculate all key metrics for the target date."""
        dau = self.metrics_calculator.calculate_dau(target_date)
        wau = self.metrics_calculator.calculate_wau(target_date)
        mau = self.metrics_calculator.calculate_mau(target_date)
        churn_rate = self.metrics_calculator.calculate_churn_rate(target_date)
        new_registrations = self.metrics_calculator.calculate_new_registrations(target_date)
        
        day_1_retention, day_7_retention, day_30_retention = self.metrics_calculator.calculate_retention_rates(target_date)
        
        return KeyMetrics(
            date=target_date,
            dau=dau,
            wau=wau,
            mau=mau,
            churn_rate=churn_rate,
            day_1_retention=day_1_retention,
            day_7_retention=day_7_retention,
            day_30_retention=day_30_retention,
            new_registrations=new_registrations,
            total_active_players=dau  # Using DAU as proxy for total active players
        )
    
    def _generate_baseline_comparisons(self, metrics: KeyMetrics, target_date: date) -> List[BaselineComparison]:
        """Generate baseline comparisons for all key metrics."""
        comparisons = []
        
        metric_mappings = [
            ('dau', metrics.dau),
            ('wau', metrics.wau),
            ('mau', metrics.mau),
            ('churn_rate', metrics.churn_rate),
            ('day_1_retention', metrics.day_1_retention),
            ('day_7_retention', metrics.day_7_retention),
            ('day_30_retention', metrics.day_30_retention),
            ('new_registrations', metrics.new_registrations)
        ]
        
        for metric_name, current_value in metric_mappings:
            comparison = self.baseline_manager.compare_with_baseline(
                metric_name, current_value, target_date
            )
            comparisons.append(comparison)
        
        return comparisons
    
    def _get_cohort_analysis(self, target_date: date) -> List[RetentionQueryResult]:
        """Get cohort analysis data for the report."""
        try:
            # Get cohort data for the past 7 days
            start_date = target_date - timedelta(days=7)
            return self.query_engine.calculate_cohort_retention(start_date, target_date)
        except Exception as e:
            logger.error(f"Error getting cohort analysis: {e}")
            return []
    
    def _generate_summary(
        self, 
        metrics: KeyMetrics, 
        comparisons: List[BaselineComparison], 
        alerts: List[RetentionAlert]
    ) -> str:
        """Generate executive summary of the report."""
        summary_parts = [
            f"Daily Retention Report for {metrics.date}",
            f"",
            f"Key Metrics:",
            f"- Daily Active Users: {metrics.dau:,}",
            f"- Weekly Active Users: {metrics.wau:,}",
            f"- Monthly Active Users: {metrics.mau:,}",
            f"- Churn Rate: {metrics.churn_rate:.1%}",
            f"- Day 1 Retention: {metrics.day_1_retention:.1%}",
            f"- Day 7 Retention: {metrics.day_7_retention:.1%}",
            f"- Day 30 Retention: {metrics.day_30_retention:.1%}",
            f"- New Registrations: {metrics.new_registrations:,}",
            f""
        ]
        
        # Add significant changes
        significant_changes = [c for c in comparisons if c.is_significant]
        if significant_changes:
            summary_parts.append("Significant Changes from Baseline:")
            for change in significant_changes:
                direction = "↑" if change.percentage_change > 0 else "↓"
                summary_parts.append(
                    f"- {change.metric_name.replace('_', ' ').title()}: "
                    f"{direction} {abs(change.percentage_change):.1f}%"
                )
            summary_parts.append("")
        
        # Add alerts summary
        if alerts:
            summary_parts.append(f"Alerts Generated: {len(alerts)}")
            critical_count = len([a for a in alerts if a.severity == AlertSeverity.CRITICAL])
            high_count = len([a for a in alerts if a.severity == AlertSeverity.HIGH])
            if critical_count > 0:
                summary_parts.append(f"- Critical: {critical_count}")
            if high_count > 0:
                summary_parts.append(f"- High: {high_count}")
        else:
            summary_parts.append("No alerts generated - all metrics within expected ranges")
        
        return "\n".join(summary_parts)
    
    def _store_report(self, report: DailyRetentionReport) -> None:
        """Store the generated report in database."""
        try:
            query = """
            INSERT INTO daily_reports (
                report_date, report_data, generated_at, created_at
            ) VALUES (
                :report_date, :report_data, :generated_at, :created_at
            )
            ON CONFLICT (report_date) 
            DO UPDATE SET 
                report_data = EXCLUDED.report_data,
                generated_at = EXCLUDED.generated_at,
                updated_at = :created_at
            """
            
            # Convert report to JSON for storage
            report_data = {
                'key_metrics': asdict(report.key_metrics),
                'baseline_comparisons': [asdict(c) for c in report.baseline_comparisons],
                'alerts': [asdict(a) for a in report.alerts],
                'cohort_analysis': [asdict(c) for c in report.cohort_analysis],
                'summary': report.summary
            }
            
            params = {
                'report_date': report.report_date,
                'report_data': json.dumps(report_data, default=str),
                'generated_at': report.generated_at,
                'created_at': datetime.now()
            }
            
            with self.db_manager.get_session() as session:
                from sqlalchemy import text
                session.execute(text(query), params)
                session.commit()
                
        except Exception as e:
            logger.error(f"Error storing report: {e}")


# Global report generator instance
report_generator = ReportGenerator()