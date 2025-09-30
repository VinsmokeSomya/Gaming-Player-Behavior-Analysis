#!/usr/bin/env python3
"""Demo script for automated reporting system.

This script demonstrates the automated reporting system functionality including:
- Daily retention summary report generation
- Key metrics calculation (DAU, WAU, MAU, churn rate)
- Historical baseline comparison
- Automated alert system for unusual retention patterns
"""

import sys
import os
from datetime import date, timedelta
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.services.reporting import (
    report_generator, 
    MetricsCalculator, 
    BaselineManager, 
    AlertManager,
    AlertSeverity
)
from src.database import db_manager
from src.config import app_config
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def demo_metrics_calculation():
    """Demonstrate key metrics calculation."""
    print("\n" + "="*60)
    print("DEMO: Key Metrics Calculation")
    print("="*60)
    
    calculator = MetricsCalculator()
    target_date = date.today() - timedelta(days=1)
    
    print(f"Calculating metrics for {target_date}...")
    
    try:
        # Calculate individual metrics
        dau = calculator.calculate_dau(target_date)
        wau = calculator.calculate_wau(target_date)
        mau = calculator.calculate_mau(target_date)
        churn_rate = calculator.calculate_churn_rate(target_date)
        new_registrations = calculator.calculate_new_registrations(target_date)
        
        day_1_retention, day_7_retention, day_30_retention = calculator.calculate_retention_rates(target_date)
        
        print(f"\nKey Metrics for {target_date}:")
        print(f"  Daily Active Users (DAU): {dau:,}")
        print(f"  Weekly Active Users (WAU): {wau:,}")
        print(f"  Monthly Active Users (MAU): {mau:,}")
        print(f"  Churn Rate: {churn_rate:.1%}")
        print(f"  Day 1 Retention: {day_1_retention:.1%}")
        print(f"  Day 7 Retention: {day_7_retention:.1%}")
        print(f"  Day 30 Retention: {day_30_retention:.1%}")
        print(f"  New Registrations: {new_registrations:,}")
        
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        logger.error(f"Metrics calculation failed: {e}")


def demo_baseline_comparison():
    """Demonstrate baseline comparison functionality."""
    print("\n" + "="*60)
    print("DEMO: Baseline Comparison")
    print("="*60)
    
    baseline_manager = BaselineManager()
    target_date = date.today() - timedelta(days=1)
    
    print(f"Comparing metrics with historical baselines for {target_date}...")
    
    try:
        # Test metrics with different scenarios
        test_metrics = [
            ('dau', 1200.0),  # Normal value
            ('churn_rate', 0.25),  # High churn rate
            ('day_7_retention', 0.20),  # Low retention
            ('new_registrations', 150)  # High registrations
        ]
        
        print(f"\nBaseline Comparisons:")
        for metric_name, current_value in test_metrics:
            comparison = baseline_manager.compare_with_baseline(
                metric_name, current_value, target_date
            )
            
            direction = "↑" if comparison.percentage_change > 0 else "↓"
            significance = " (SIGNIFICANT)" if comparison.is_significant else ""
            threshold = " [THRESHOLD BREACHED]" if comparison.threshold_breached else ""
            
            print(f"  {metric_name.replace('_', ' ').title()}:")
            print(f"    Current: {comparison.current_value:.2f}")
            print(f"    Baseline: {comparison.baseline_value:.2f}")
            print(f"    Change: {direction} {abs(comparison.percentage_change):.1f}%{significance}{threshold}")
        
    except Exception as e:
        print(f"Error in baseline comparison: {e}")
        logger.error(f"Baseline comparison failed: {e}")


def demo_alert_generation():
    """Demonstrate alert generation functionality."""
    print("\n" + "="*60)
    print("DEMO: Alert Generation")
    print("="*60)
    
    alert_manager = AlertManager()
    
    print("Generating sample alerts for unusual patterns...")
    
    try:
        # Create sample metrics with concerning values
        from src.services.reporting import KeyMetrics, BaselineComparison
        
        sample_metrics = KeyMetrics(
            date=date.today() - timedelta(days=1),
            dau=800,  # Lower than expected
            wau=2500,
            mau=8000,
            churn_rate=0.35,  # High churn rate
            day_1_retention=0.45,
            day_7_retention=0.20,  # Low retention
            day_30_retention=0.08,
            new_registrations=45,
            total_active_players=800
        )
        
        # Create sample baseline comparisons with threshold breaches
        sample_comparisons = [
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
            ),
            BaselineComparison(
                metric_name="day_7_retention",
                current_value=0.20,
                baseline_value=0.35,
                percentage_change=-42.9,
                is_significant=True,
                threshold_breached=True
            )
        ]
        
        # Generate alerts
        alerts = alert_manager.generate_alerts(sample_metrics, sample_comparisons)
        
        print(f"\nGenerated {len(alerts)} alerts:")
        for alert in alerts:
            severity_color = {
                AlertSeverity.CRITICAL: "🔴",
                AlertSeverity.HIGH: "🟠", 
                AlertSeverity.MEDIUM: "🟡",
                AlertSeverity.LOW: "🟢"
            }
            
            print(f"  {severity_color.get(alert.severity, '⚪')} {alert.severity.value.upper()}: {alert.message}")
            print(f"    Metric: {alert.metric_name}")
            print(f"    Current: {alert.current_value:.2f}, Expected: {alert.expected_value:.2f}")
            print(f"    Deviation: {alert.deviation_percentage:.1f}%")
            print()
        
        # Store alerts (in a real scenario)
        print("Storing alerts in database...")
        alert_manager.store_alerts(alerts)
        
        # Simulate sending notifications
        print("Sending alert notifications...")
        alert_manager.send_alert_notifications(alerts)
        
    except Exception as e:
        print(f"Error in alert generation: {e}")
        logger.error(f"Alert generation failed: {e}")


def demo_daily_report_generation():
    """Demonstrate complete daily report generation."""
    print("\n" + "="*60)
    print("DEMO: Daily Report Generation")
    print("="*60)
    
    target_date = date.today() - timedelta(days=1)
    print(f"Generating comprehensive daily report for {target_date}...")
    
    try:
        # Generate the complete daily report
        report = report_generator.generate_daily_report(target_date)
        
        print(f"\n📊 DAILY RETENTION REPORT")
        print(f"Report Date: {report.report_date}")
        print(f"Generated At: {report.generated_at}")
        print("\n" + "-"*50)
        print(report.summary)
        print("-"*50)
        
        # Display key metrics
        metrics = report.key_metrics
        print(f"\n📈 KEY METRICS:")
        print(f"  DAU: {metrics.dau:,}")
        print(f"  WAU: {metrics.wau:,}")
        print(f"  MAU: {metrics.mau:,}")
        print(f"  Churn Rate: {metrics.churn_rate:.1%}")
        print(f"  Day 1 Retention: {metrics.day_1_retention:.1%}")
        print(f"  Day 7 Retention: {metrics.day_7_retention:.1%}")
        print(f"  Day 30 Retention: {metrics.day_30_retention:.1%}")
        print(f"  New Registrations: {metrics.new_registrations:,}")
        
        # Display significant baseline comparisons
        significant_comparisons = [c for c in report.baseline_comparisons if c.is_significant]
        if significant_comparisons:
            print(f"\n📊 SIGNIFICANT CHANGES:")
            for comp in significant_comparisons:
                direction = "↑" if comp.percentage_change > 0 else "↓"
                print(f"  {comp.metric_name.replace('_', ' ').title()}: {direction} {abs(comp.percentage_change):.1f}%")
        
        # Display alerts
        if report.alerts:
            print(f"\n🚨 ALERTS ({len(report.alerts)}):")
            for alert in report.alerts:
                severity_icon = {
                    AlertSeverity.CRITICAL: "🔴",
                    AlertSeverity.HIGH: "🟠",
                    AlertSeverity.MEDIUM: "🟡", 
                    AlertSeverity.LOW: "🟢"
                }
                print(f"  {severity_icon.get(alert.severity, '⚪')} {alert.message}")
        else:
            print(f"\n✅ No alerts - all metrics within expected ranges")
        
        # Display cohort analysis summary
        if report.cohort_analysis:
            print(f"\n👥 COHORT ANALYSIS:")
            print(f"  Analyzed {len(report.cohort_analysis)} cohorts")
            avg_day_1 = sum(c.day_1_retention for c in report.cohort_analysis) / len(report.cohort_analysis)
            avg_day_7 = sum(c.day_7_retention for c in report.cohort_analysis) / len(report.cohort_analysis)
            print(f"  Average Day 1 Retention: {avg_day_1:.1%}")
            print(f"  Average Day 7 Retention: {avg_day_7:.1%}")
        
        print(f"\n✅ Report generated and stored successfully!")
        
    except Exception as e:
        print(f"Error generating daily report: {e}")
        logger.error(f"Daily report generation failed: {e}")


def demo_historical_reports():
    """Demonstrate accessing historical reports."""
    print("\n" + "="*60)
    print("DEMO: Historical Reports Access")
    print("="*60)
    
    print("Accessing stored historical reports...")
    
    try:
        # Query for recent reports
        query = """
        SELECT report_date, generated_at, 
               report_data->>'summary' as summary
        FROM daily_reports 
        ORDER BY report_date DESC 
        LIMIT 5
        """
        
        with db_manager.get_session() as session:
            from sqlalchemy import text
            result = session.execute(text(query))
            reports = result.fetchall()
            
            if reports:
                print(f"\nFound {len(reports)} recent reports:")
                for report in reports:
                    print(f"  📅 {report.report_date} (Generated: {report.generated_at})")
                    if report.summary:
                        # Show first line of summary
                        first_line = report.summary.split('\n')[0] if report.summary else "No summary"
                        print(f"     {first_line}")
            else:
                print("No historical reports found. Generate some reports first!")
        
    except Exception as e:
        print(f"Error accessing historical reports: {e}")
        logger.error(f"Historical reports access failed: {e}")


def main():
    """Run all reporting system demos."""
    print("🚀 PLAYER RETENTION ANALYTICS - AUTOMATED REPORTING SYSTEM DEMO")
    print("="*80)
    
    # Initialize database connection
    try:
        if not db_manager.wait_for_connection():
            print("❌ Could not connect to database. Please ensure PostgreSQL is running.")
            return
        
        print("✅ Database connection established")
        
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        return
    
    # Run all demos
    try:
        demo_metrics_calculation()
        demo_baseline_comparison()
        demo_alert_generation()
        demo_daily_report_generation()
        demo_historical_reports()
        
        print("\n" + "="*80)
        print("🎉 All reporting system demos completed successfully!")
        print("="*80)
        
        print("\nNext Steps:")
        print("1. Set up automated daily report generation (cron job or scheduler)")
        print("2. Configure email/Slack notifications for alerts")
        print("3. Create dashboard views for report visualization")
        print("4. Set up data retention policies for historical data")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        logger.error(f"Demo execution failed: {e}")
    
    finally:
        # Clean up database connections
        db_manager.close()


if __name__ == "__main__":
    main()