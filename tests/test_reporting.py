"""Unit tests for automated reporting system.

Tests cover:
- Daily retention summary report generation
- Key metrics calculation functions (DAU, WAU, MAU, churn rate)
- Historical baseline comparison logic
- Automated alert system for unusual retention patterns
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import date, datetime, timedelta
from decimal import Decimal

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from services.reporting import (
    MetricsCalculator,
    BaselineManager,
    AlertManager,
    ReportGenerator,
    KeyMetrics,
    BaselineComparison,
    RetentionAlert,
    AlertSeverity,
    DailyRetentionReport
)
from analytics.query_engine import RetentionQueryResult


class TestMetricsCalculator:
    """Test suite for MetricsCalculator class."""
    
    @pytest.fixture
    def calculator(self):
        """Create MetricsCalculator instance with mocked dependencies."""
        with patch('services.reporting.query_engine') as mock_query_engine, \
             patch('services.reporting.db_manager') as mock_db_manager:
            calculator = MetricsCalculator()
            calculator.query_engine = mock_query_engine
            calculator.db_manager = mock_db_manager
            return calculator
    
    def test_calculate_dau_success(self, calculator):
        """Test successful DAU calculation."""
        target_date = date(2024, 1, 15)
        calculator.query_engine.get_daily_active_users.return_value = [(target_date, 1250)]
        
        result = calculator.calculate_dau(target_date)
        
        assert result == 1250
        calculator.query_engine.get_daily_active_users.assert_called_once_with(target_date, target_date)
    
    def test_calculate_dau_no_data(self, calculator):
        """Test DAU calculation with no data."""
        target_date = date(2024, 1, 15)
        calculator.query_engine.get_daily_active_users.return_value = []
        
        result = calculator.calculate_dau(target_date)
        
        assert result == 0
    
    def test_calculate_dau_error(self, calculator):
        """Test DAU calculation with database error."""
        target_date = date(2024, 1, 15)
        calculator.query_engine.get_daily_active_users.side_effect = Exception("DB Error")
        
        result = calculator.calculate_dau(target_date)
        
        assert result == 0
    
    def test_calculate_wau_success(self, calculator):
        """Test successful WAU calculation."""
        target_date = date(2024, 1, 15)  # Monday
        week_start = date(2024, 1, 15)
        week_end = date(2024, 1, 21)
        calculator.query_engine.get_weekly_active_users.return_value = [(week_start, 4200)]
        
        result = calculator.calculate_wau(target_date)
        
        assert result == 4200
        calculator.query_engine.get_weekly_active_users.assert_called_once_with(week_start, week_end)
    
    def test_calculate_mau_success(self, calculator):
        """Test successful MAU calculation."""
        target_date = date(2024, 1, 15)
        
        # Mock database session and query result
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_row = MagicMock()
        mock_row.mau_count = 12500
        mock_result.fetchone.return_value = mock_row
        mock_session.execute.return_value = mock_result
        
        calculator.db_manager.get_session.return_value.__enter__.return_value = mock_session
        
        result = calculator.calculate_mau(target_date)
        
        assert result == 12500
        mock_session.execute.assert_called_once()
    
    def test_calculate_churn_rate_success(self, calculator):
        """Test successful churn rate calculation."""
        target_date = date(2024, 1, 15)
        
        # Mock database session and query result
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_row = MagicMock()
        mock_row.churn_rate = Decimal('0.15')
        mock_result.fetchone.return_value = mock_row
        mock_session.execute.return_value = mock_result
        
        calculator.db_manager.get_session.return_value.__enter__.return_value = mock_session
        
        result = calculator.calculate_churn_rate(target_date)
        
        assert result == 0.15
    
    def test_calculate_new_registrations_success(self, calculator):
        """Test successful new registrations calculation."""
        target_date = date(2024, 1, 15)
        
        # Mock database session and query result
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_row = MagicMock()
        mock_row.new_registrations = 85
        mock_result.fetchone.return_value = mock_row
        mock_session.execute.return_value = mock_result
        
        calculator.db_manager.get_session.return_value.__enter__.return_value = mock_session
        
        result = calculator.calculate_new_registrations(target_date)
        
        assert result == 85
    
    def test_calculate_retention_rates_success(self, calculator):
        """Test successful retention rates calculation."""
        target_date = date(2024, 1, 15)
        
        # Mock retention query results
        retention_data = [
            RetentionQueryResult(
                cohort_date=date(2024, 1, 14),
                day_1_retention=0.65,
                day_7_retention=0.35,
                day_30_retention=0.18,
                cohort_size=100
            ),
            RetentionQueryResult(
                cohort_date=date(2024, 1, 13),
                day_1_retention=0.70,
                day_7_retention=0.40,
                day_30_retention=0.20,
                cohort_size=150
            )
        ]
        
        calculator.query_engine.calculate_cohort_retention.return_value = retention_data
        
        day_1, day_7, day_30 = calculator.calculate_retention_rates(target_date)
        
        # Expected weighted averages
        expected_day_1 = (0.65 * 100 + 0.70 * 150) / 250  # 0.68
        expected_day_7 = (0.35 * 100 + 0.40 * 150) / 250  # 0.38
        expected_day_30 = (0.18 * 100 + 0.20 * 150) / 250  # 0.192
        
        assert abs(day_1 - expected_day_1) < 0.01
        assert abs(day_7 - expected_day_7) < 0.01
        assert abs(day_30 - expected_day_30) < 0.01
    
    def test_calculate_retention_rates_no_data(self, calculator):
        """Test retention rates calculation with no data."""
        target_date = date(2024, 1, 15)
        calculator.query_engine.calculate_cohort_retention.return_value = []
        
        day_1, day_7, day_30 = calculator.calculate_retention_rates(target_date)
        
        assert day_1 == 0.0
        assert day_7 == 0.0
        assert day_30 == 0.0


class TestBaselineManager:
    """Test suite for BaselineManager class."""
    
    @pytest.fixture
    def baseline_manager(self):
        """Create BaselineManager instance with mocked dependencies."""
        with patch('services.reporting.db_manager') as mock_db_manager:
            manager = BaselineManager()
            manager.db_manager = mock_db_manager
            return manager
    
    def test_calculate_baseline_success(self, baseline_manager):
        """Test successful baseline calculation."""
        target_date = date(2024, 1, 15)
        
        # Mock database session and query result
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_row = MagicMock()
        mock_row.baseline_value = Decimal('1200.5')
        mock_result.fetchone.return_value = mock_row
        mock_session.execute.return_value = mock_result
        
        baseline_manager.db_manager.get_session.return_value.__enter__.return_value = mock_session
        
        result = baseline_manager.calculate_baseline('dau', target_date)
        
        assert result == 1200.5
    
    def test_calculate_baseline_no_data(self, baseline_manager):
        """Test baseline calculation with no historical data."""
        target_date = date(2024, 1, 15)
        
        # Mock database session with no results
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_result.fetchone.return_value = None
        mock_session.execute.return_value = mock_result
        
        baseline_manager.db_manager.get_session.return_value.__enter__.return_value = mock_session
        
        result = baseline_manager.calculate_baseline('dau', target_date)
        
        assert result == 0.0
    
    def test_store_daily_metrics_success(self, baseline_manager):
        """Test successful daily metrics storage."""
        metrics = KeyMetrics(
            date=date(2024, 1, 15),
            dau=1250,
            wau=4200,
            mau=12500,
            churn_rate=0.15,
            day_1_retention=0.65,
            day_7_retention=0.35,
            day_30_retention=0.18,
            new_registrations=85,
            total_active_players=1250
        )
        
        # Mock database session
        mock_session = MagicMock()
        baseline_manager.db_manager.get_session.return_value.__enter__.return_value = mock_session
        
        baseline_manager.store_daily_metrics(metrics)
        
        # Verify that execute was called for each metric
        assert mock_session.execute.call_count == 9  # 9 metrics stored
        mock_session.commit.assert_called_once()
    
    def test_compare_with_baseline_significant_decrease(self, baseline_manager):
        """Test baseline comparison with significant decrease."""
        target_date = date(2024, 1, 15)
        
        # Mock baseline calculation
        baseline_manager.calculate_baseline = Mock(return_value=1000.0)
        
        comparison = baseline_manager.compare_with_baseline('dau', 750.0, target_date)
        
        assert comparison.metric_name == 'dau'
        assert comparison.current_value == 750.0
        assert comparison.baseline_value == 1000.0
        assert comparison.percentage_change == -25.0
        assert comparison.is_significant is True  # > 10% change
        assert comparison.threshold_breached is True  # > 20% decrease
    
    def test_compare_with_baseline_churn_increase(self, baseline_manager):
        """Test baseline comparison with churn rate increase."""
        target_date = date(2024, 1, 15)
        
        # Mock baseline calculation
        baseline_manager.calculate_baseline = Mock(return_value=0.15)
        
        comparison = baseline_manager.compare_with_baseline('churn_rate', 0.20, target_date)
        
        assert comparison.metric_name == 'churn_rate'
        assert comparison.current_value == 0.20
        assert comparison.baseline_value == 0.15
        assert comparison.percentage_change == pytest.approx(33.33, rel=0.01)
        assert comparison.is_significant is True
        assert comparison.threshold_breached is True  # > 25% increase in churn


class TestAlertManager:
    """Test suite for AlertManager class."""
    
    @pytest.fixture
    def alert_manager(self):
        """Create AlertManager instance with mocked dependencies."""
        with patch('services.reporting.db_manager') as mock_db_manager:
            manager = AlertManager()
            manager.db_manager = mock_db_manager
            return manager
    
    def test_generate_alerts_threshold_breach(self, alert_manager):
        """Test alert generation for threshold breaches."""
        metrics = KeyMetrics(
            date=date(2024, 1, 15),
            dau=800,  # Low DAU
            wau=2500,
            mau=8000,
            churn_rate=0.35,  # High churn
            day_1_retention=0.45,
            day_7_retention=0.20,
            day_30_retention=0.08,
            new_registrations=45,
            total_active_players=800
        )
        
        comparisons = [
            BaselineComparison(
                metric_name="dau",
                current_value=800.0,
                baseline_value=1200.0,
                percentage_change=-33.3,
                is_significant=True,
                threshold_breached=True
            ),
            BaselineComparison(
                metric_name="churn_rate",
                current_value=0.35,
                baseline_value=0.15,
                percentage_change=133.3,
                is_significant=True,
                threshold_breached=True
            )
        ]
        
        alerts = alert_manager.generate_alerts(metrics, comparisons)
        
        # Should generate alerts for threshold breaches plus custom alerts
        assert len(alerts) >= 2
        
        # Check for DAU alert
        dau_alerts = [a for a in alerts if a.metric_name == 'dau']
        assert len(dau_alerts) >= 1
        
        # Check for churn rate alerts
        churn_alerts = [a for a in alerts if a.metric_name == 'churn_rate']
        assert len(churn_alerts) >= 1
        
        # Verify alert properties
        for alert in alerts:
            assert alert.alert_id is not None
            assert alert.severity in [AlertSeverity.LOW, AlertSeverity.MEDIUM, AlertSeverity.HIGH, AlertSeverity.CRITICAL]
            assert alert.message is not None
            assert alert.timestamp is not None
    
    def test_determine_severity_critical(self, alert_manager):
        """Test severity determination for critical changes."""
        comparison = BaselineComparison(
            metric_name="dau",
            current_value=500.0,
            baseline_value=1000.0,
            percentage_change=-50.0,
            is_significant=True,
            threshold_breached=True
        )
        
        severity = alert_manager._determine_severity(comparison)
        
        assert severity == AlertSeverity.CRITICAL
    
    def test_determine_severity_high(self, alert_manager):
        """Test severity determination for high changes."""
        comparison = BaselineComparison(
            metric_name="dau",
            current_value=700.0,
            baseline_value=1000.0,
            percentage_change=-30.0,
            is_significant=True,
            threshold_breached=True
        )
        
        severity = alert_manager._determine_severity(comparison)
        
        assert severity == AlertSeverity.HIGH
    
    def test_generate_alert_message(self, alert_manager):
        """Test alert message generation."""
        comparison = BaselineComparison(
            metric_name="day_7_retention",
            current_value=0.25,
            baseline_value=0.35,
            percentage_change=-28.6,
            is_significant=True,
            threshold_breached=True
        )
        
        message = alert_manager._generate_alert_message(comparison)
        
        assert "Day 7 Retention" in message
        assert "decreased" in message
        assert "28.6%" in message
        assert "0.35" in message
        assert "0.25" in message
    
    def test_store_alerts_success(self, alert_manager):
        """Test successful alert storage."""
        alerts = [
            RetentionAlert(
                alert_id="test_alert_1",
                severity=AlertSeverity.HIGH,
                metric_name="dau",
                current_value=800.0,
                expected_value=1200.0,
                deviation_percentage=-33.3,
                message="Test alert message",
                timestamp=datetime.now()
            )
        ]
        
        # Mock database session
        mock_session = MagicMock()
        alert_manager.db_manager.get_session.return_value.__enter__.return_value = mock_session
        
        alert_manager.store_alerts(alerts)
        
        mock_session.execute.assert_called_once()
        mock_session.commit.assert_called_once()
    
    def test_store_alerts_empty_list(self, alert_manager):
        """Test storing empty alerts list."""
        alert_manager.store_alerts([])
        
        # Should not interact with database
        alert_manager.db_manager.get_session.assert_not_called()


class TestReportGenerator:
    """Test suite for ReportGenerator class."""
    
    @pytest.fixture
    def report_generator(self):
        """Create ReportGenerator instance with mocked dependencies."""
        with patch('services.reporting.MetricsCalculator') as mock_calc, \
             patch('services.reporting.BaselineManager') as mock_baseline, \
             patch('services.reporting.AlertManager') as mock_alert, \
             patch('services.reporting.query_engine') as mock_query:
            
            generator = ReportGenerator()
            generator.metrics_calculator = mock_calc.return_value
            generator.baseline_manager = mock_baseline.return_value
            generator.alert_manager = mock_alert.return_value
            generator.query_engine = mock_query
            
            return generator
    
    def test_generate_daily_report_success(self, report_generator):
        """Test successful daily report generation."""
        target_date = date(2024, 1, 15)
        
        # Mock key metrics calculation
        sample_metrics = KeyMetrics(
            date=target_date,
            dau=1250,
            wau=4200,
            mau=12500,
            churn_rate=0.15,
            day_1_retention=0.65,
            day_7_retention=0.35,
            day_30_retention=0.18,
            new_registrations=85,
            total_active_players=1250
        )
        
        report_generator._calculate_key_metrics = Mock(return_value=sample_metrics)
        report_generator.baseline_manager.store_daily_metrics = Mock()
        report_generator._generate_baseline_comparisons = Mock(return_value=[])
        report_generator.alert_manager.generate_alerts = Mock(return_value=[])
        report_generator.alert_manager.store_alerts = Mock()
        report_generator.alert_manager.send_alert_notifications = Mock()
        report_generator._get_cohort_analysis = Mock(return_value=[])
        report_generator._generate_summary = Mock(return_value="Test summary")
        report_generator._store_report = Mock()
        
        report = report_generator.generate_daily_report(target_date)
        
        assert isinstance(report, DailyRetentionReport)
        assert report.report_date == target_date
        assert report.key_metrics == sample_metrics
        assert report.summary == "Test summary"
        assert report.generated_at is not None
        
        # Verify all methods were called
        report_generator._calculate_key_metrics.assert_called_once_with(target_date)
        report_generator.baseline_manager.store_daily_metrics.assert_called_once_with(sample_metrics)
        report_generator._store_report.assert_called_once()
    
    def test_calculate_key_metrics(self, report_generator):
        """Test key metrics calculation."""
        target_date = date(2024, 1, 15)
        
        # Mock individual metric calculations
        report_generator.metrics_calculator.calculate_dau.return_value = 1250
        report_generator.metrics_calculator.calculate_wau.return_value = 4200
        report_generator.metrics_calculator.calculate_mau.return_value = 12500
        report_generator.metrics_calculator.calculate_churn_rate.return_value = 0.15
        report_generator.metrics_calculator.calculate_new_registrations.return_value = 85
        report_generator.metrics_calculator.calculate_retention_rates.return_value = (0.65, 0.35, 0.18)
        
        metrics = report_generator._calculate_key_metrics(target_date)
        
        assert metrics.date == target_date
        assert metrics.dau == 1250
        assert metrics.wau == 4200
        assert metrics.mau == 12500
        assert metrics.churn_rate == 0.15
        assert metrics.day_1_retention == 0.65
        assert metrics.day_7_retention == 0.35
        assert metrics.day_30_retention == 0.18
        assert metrics.new_registrations == 85
        assert metrics.total_active_players == 1250
    
    def test_generate_baseline_comparisons(self, report_generator):
        """Test baseline comparisons generation."""
        target_date = date(2024, 1, 15)
        sample_metrics = KeyMetrics(
            date=target_date,
            dau=1250,
            wau=4200,
            mau=12500,
            churn_rate=0.15,
            day_1_retention=0.65,
            day_7_retention=0.35,
            day_30_retention=0.18,
            new_registrations=85,
            total_active_players=1250
        )
        
        # Mock baseline comparison
        sample_comparison = BaselineComparison(
            metric_name="dau",
            current_value=1250.0,
            baseline_value=1200.0,
            percentage_change=4.17,
            is_significant=False,
            threshold_breached=False
        )
        
        report_generator.baseline_manager.compare_with_baseline.return_value = sample_comparison
        
        comparisons = report_generator._generate_baseline_comparisons(sample_metrics, target_date)
        
        # Should generate comparisons for 8 metrics
        assert len(comparisons) == 8
        assert all(isinstance(c, BaselineComparison) for c in comparisons)
    
    def test_generate_summary_with_alerts(self, report_generator):
        """Test summary generation with alerts."""
        sample_metrics = KeyMetrics(
            date=date(2024, 1, 15),
            dau=1250,
            wau=4200,
            mau=12500,
            churn_rate=0.15,
            day_1_retention=0.65,
            day_7_retention=0.35,
            day_30_retention=0.18,
            new_registrations=85,
            total_active_players=1250
        )
        
        sample_comparisons = [
            BaselineComparison(
                metric_name="dau",
                current_value=1250.0,
                baseline_value=1000.0,
                percentage_change=25.0,
                is_significant=True,
                threshold_breached=False
            )
        ]
        
        sample_alerts = [
            RetentionAlert(
                alert_id="test_alert",
                severity=AlertSeverity.HIGH,
                metric_name="churn_rate",
                current_value=0.25,
                expected_value=0.15,
                deviation_percentage=66.7,
                message="Test alert",
                timestamp=datetime.now()
            )
        ]
        
        summary = report_generator._generate_summary(sample_metrics, sample_comparisons, sample_alerts)
        
        assert "Daily Retention Report" in summary
        assert "1,250" in summary  # DAU formatting
        assert "15.0%" in summary  # Churn rate formatting
        assert "Significant Changes" in summary
        assert "Alerts Generated: 1" in summary
        assert "High: 1" in summary


class TestIntegration:
    """Integration tests for the complete reporting system."""
    
    @pytest.fixture
    def mock_db_data(self):
        """Mock database data for integration tests."""
        return {
            'dau_data': [(date(2024, 1, 15), 1250)],
            'wau_data': [(date(2024, 1, 15), 4200)],
            'retention_data': [
                RetentionQueryResult(
                    cohort_date=date(2024, 1, 14),
                    day_1_retention=0.65,
                    day_7_retention=0.35,
                    day_30_retention=0.18,
                    cohort_size=100
                )
            ]
        }
    
    def test_end_to_end_report_generation(self, mock_db_data):
        """Test end-to-end report generation process."""
        target_date = date(2024, 1, 15)
        
        # Create a fully mocked report generator
        with patch('services.reporting.MetricsCalculator') as mock_calc_class, \
             patch('services.reporting.BaselineManager') as mock_baseline_class, \
             patch('services.reporting.AlertManager') as mock_alert_class, \
             patch('services.reporting.query_engine') as mock_query_engine:
            
            # Mock the calculator methods
            mock_calc = mock_calc_class.return_value
            mock_calc.calculate_dau.return_value = 1250
            mock_calc.calculate_wau.return_value = 4200
            mock_calc.calculate_mau.return_value = 12500
            mock_calc.calculate_churn_rate.return_value = 0.15
            mock_calc.calculate_new_registrations.return_value = 85
            mock_calc.calculate_retention_rates.return_value = (0.65, 0.35, 0.18)
            
            # Mock the baseline manager
            mock_baseline = mock_baseline_class.return_value
            mock_baseline.store_daily_metrics = Mock()
            mock_baseline.compare_with_baseline.return_value = BaselineComparison(
                metric_name="dau",
                current_value=1250.0,
                baseline_value=1200.0,
                percentage_change=4.17,
                is_significant=False,
                threshold_breached=False
            )
            
            # Mock the alert manager
            mock_alert = mock_alert_class.return_value
            mock_alert.generate_alerts.return_value = []
            mock_alert.store_alerts = Mock()
            mock_alert.send_alert_notifications = Mock()
            
            # Mock query engine
            mock_query_engine.calculate_cohort_retention.return_value = mock_db_data['retention_data']
            
            # Create report generator and generate report
            generator = ReportGenerator()
            
            # Mock the _store_report method to avoid database interaction
            generator._store_report = Mock()
            
            report = generator.generate_daily_report(target_date)
            
            # Verify report structure
            assert isinstance(report, DailyRetentionReport)
            assert report.report_date == target_date
            assert isinstance(report.key_metrics, KeyMetrics)
            assert isinstance(report.baseline_comparisons, list)
            assert isinstance(report.alerts, list)
            assert isinstance(report.summary, str)
            
            # Verify key metrics
            assert report.key_metrics.dau == 1250
            assert report.key_metrics.wau == 4200
            assert report.key_metrics.mau == 12500
            assert report.key_metrics.churn_rate == 0.15
            assert report.key_metrics.new_registrations == 85
            assert report.key_metrics.date == target_date


if __name__ == "__main__":
    pytest.main([__file__, "-v"])