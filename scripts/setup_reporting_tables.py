#!/usr/bin/env python3
"""Setup script for reporting system database tables.

This script creates the necessary database tables for the automated reporting system.
"""

import sys
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database import db_manager
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_reporting_tables():
    """Create reporting system database tables."""
    
    # SQL to create reporting tables
    create_tables_sql = """
    -- Table for storing daily metrics history for baseline calculations
    CREATE TABLE IF NOT EXISTS daily_metrics_history (
        id SERIAL PRIMARY KEY,
        metric_date DATE NOT NULL,
        metric_name VARCHAR(50) NOT NULL,
        metric_value DECIMAL(10, 4) NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(metric_date, metric_name)
    );

    -- Index for efficient baseline calculations
    CREATE INDEX IF NOT EXISTS idx_daily_metrics_history_date_name 
    ON daily_metrics_history(metric_date, metric_name);

    -- Table for storing retention alerts
    CREATE TABLE IF NOT EXISTS retention_alerts (
        id SERIAL PRIMARY KEY,
        alert_id VARCHAR(100) UNIQUE NOT NULL,
        severity VARCHAR(20) NOT NULL CHECK (severity IN ('low', 'medium', 'high', 'critical')),
        metric_name VARCHAR(50) NOT NULL,
        current_value DECIMAL(10, 4) NOT NULL,
        expected_value DECIMAL(10, 4) NOT NULL,
        deviation_percentage DECIMAL(8, 2) NOT NULL,
        message TEXT NOT NULL,
        timestamp TIMESTAMP NOT NULL,
        resolved BOOLEAN DEFAULT FALSE,
        resolved_at TIMESTAMP NULL,
        resolved_by VARCHAR(100) NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    -- Indexes for alert management
    CREATE INDEX IF NOT EXISTS idx_retention_alerts_timestamp 
    ON retention_alerts(timestamp DESC);

    CREATE INDEX IF NOT EXISTS idx_retention_alerts_severity 
    ON retention_alerts(severity);

    CREATE INDEX IF NOT EXISTS idx_retention_alerts_resolved 
    ON retention_alerts(resolved);

    -- Table for storing daily reports
    CREATE TABLE IF NOT EXISTS daily_reports (
        id SERIAL PRIMARY KEY,
        report_date DATE UNIQUE NOT NULL,
        report_data JSONB NOT NULL,
        generated_at TIMESTAMP NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    -- Index for efficient report retrieval
    CREATE INDEX IF NOT EXISTS idx_daily_reports_date 
    ON daily_reports(report_date DESC);

    -- GIN index for JSONB queries on report data
    CREATE INDEX IF NOT EXISTS idx_daily_reports_data 
    ON daily_reports USING GIN (report_data);

    -- Table for storing alert notification history
    CREATE TABLE IF NOT EXISTS alert_notifications (
        id SERIAL PRIMARY KEY,
        alert_id VARCHAR(100) NOT NULL,
        notification_type VARCHAR(50) NOT NULL,
        recipient VARCHAR(200) NOT NULL,
        status VARCHAR(20) NOT NULL CHECK (status IN ('pending', 'sent', 'failed', 'bounced')),
        sent_at TIMESTAMP NULL,
        error_message TEXT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    -- Index for notification tracking
    CREATE INDEX IF NOT EXISTS idx_alert_notifications_alert_id 
    ON alert_notifications(alert_id);

    CREATE INDEX IF NOT EXISTS idx_alert_notifications_status 
    ON alert_notifications(status);
    """
    
    # SQL to create views
    create_views_sql = """
    -- View for recent alerts summary
    CREATE OR REPLACE VIEW recent_alerts_summary AS
    SELECT 
        severity,
        COUNT(*) as alert_count,
        COUNT(CASE WHEN resolved = FALSE THEN 1 END) as unresolved_count,
        MAX(timestamp) as latest_alert
    FROM retention_alerts 
    WHERE timestamp >= CURRENT_DATE - INTERVAL '7 days'
    GROUP BY severity
    ORDER BY 
        CASE severity
            WHEN 'critical' THEN 1
            WHEN 'high' THEN 2
            WHEN 'medium' THEN 3
            WHEN 'low' THEN 4
        END;

    -- View for metrics trends
    CREATE OR REPLACE VIEW metrics_trends AS
    SELECT 
        metric_name,
        metric_date,
        metric_value,
        LAG(metric_value, 1) OVER (PARTITION BY metric_name ORDER BY metric_date) as previous_value,
        LAG(metric_value, 7) OVER (PARTITION BY metric_name ORDER BY metric_date) as week_ago_value,
        LAG(metric_value, 30) OVER (PARTITION BY metric_name ORDER BY metric_date) as month_ago_value
    FROM daily_metrics_history
    WHERE metric_date >= CURRENT_DATE - INTERVAL '90 days'
    ORDER BY metric_name, metric_date DESC;
    """
    
    # SQL to create triggers
    create_triggers_sql = """
    -- Function to automatically update updated_at timestamp
    CREATE OR REPLACE FUNCTION update_updated_at_column()
    RETURNS TRIGGER AS $$
    BEGIN
        NEW.updated_at = CURRENT_TIMESTAMP;
        RETURN NEW;
    END;
    $$ language 'plpgsql';

    -- Triggers for automatic updated_at updates
    DROP TRIGGER IF EXISTS update_daily_metrics_history_updated_at ON daily_metrics_history;
    CREATE TRIGGER update_daily_metrics_history_updated_at 
        BEFORE UPDATE ON daily_metrics_history 
        FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

    DROP TRIGGER IF EXISTS update_retention_alerts_updated_at ON retention_alerts;
    CREATE TRIGGER update_retention_alerts_updated_at 
        BEFORE UPDATE ON retention_alerts 
        FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

    DROP TRIGGER IF EXISTS update_daily_reports_updated_at ON daily_reports;
    CREATE TRIGGER update_daily_reports_updated_at 
        BEFORE UPDATE ON daily_reports 
        FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

    DROP TRIGGER IF EXISTS update_alert_notifications_updated_at ON alert_notifications;
    CREATE TRIGGER update_alert_notifications_updated_at 
        BEFORE UPDATE ON alert_notifications 
        FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    """
    
    # SQL to insert sample data
    insert_sample_data_sql = """
    -- Insert some sample baseline data for testing
    INSERT INTO daily_metrics_history (metric_date, metric_name, metric_value) VALUES
        (CURRENT_DATE - INTERVAL '30 days', 'dau', 1250),
        (CURRENT_DATE - INTERVAL '30 days', 'wau', 4200),
        (CURRENT_DATE - INTERVAL '30 days', 'mau', 12500),
        (CURRENT_DATE - INTERVAL '30 days', 'churn_rate', 0.15),
        (CURRENT_DATE - INTERVAL '30 days', 'day_1_retention', 0.65),
        (CURRENT_DATE - INTERVAL '30 days', 'day_7_retention', 0.35),
        (CURRENT_DATE - INTERVAL '30 days', 'day_30_retention', 0.18),
        (CURRENT_DATE - INTERVAL '30 days', 'new_registrations', 85),
        
        (CURRENT_DATE - INTERVAL '29 days', 'dau', 1180),
        (CURRENT_DATE - INTERVAL '29 days', 'wau', 4150),
        (CURRENT_DATE - INTERVAL '29 days', 'mau', 12400),
        (CURRENT_DATE - INTERVAL '29 days', 'churn_rate', 0.16),
        (CURRENT_DATE - INTERVAL '29 days', 'day_1_retention', 0.63),
        (CURRENT_DATE - INTERVAL '29 days', 'day_7_retention', 0.34),
        (CURRENT_DATE - INTERVAL '29 days', 'day_30_retention', 0.17),
        (CURRENT_DATE - INTERVAL '29 days', 'new_registrations', 92),
        
        (CURRENT_DATE - INTERVAL '28 days', 'dau', 1320),
        (CURRENT_DATE - INTERVAL '28 days', 'wau', 4300),
        (CURRENT_DATE - INTERVAL '28 days', 'mau', 12600),
        (CURRENT_DATE - INTERVAL '28 days', 'churn_rate', 0.14),
        (CURRENT_DATE - INTERVAL '28 days', 'day_1_retention', 0.67),
        (CURRENT_DATE - INTERVAL '28 days', 'day_7_retention', 0.36),
        (CURRENT_DATE - INTERVAL '28 days', 'day_30_retention', 0.19),
        (CURRENT_DATE - INTERVAL '28 days', 'new_registrations', 78)
    ON CONFLICT (metric_date, metric_name) DO NOTHING;
    """
    
    try:
        print("🚀 Setting up reporting system database tables...")
        
        # Initialize database connection
        if not db_manager.wait_for_connection():
            print("❌ Could not connect to database. Please ensure PostgreSQL is running.")
            return False
        
        print("✅ Database connection established")
        
        # Execute table creation
        print("📊 Creating tables...")
        with db_manager.get_session() as session:
            from sqlalchemy import text
            session.execute(text(create_tables_sql))
            session.commit()
        print("✅ Tables created successfully")
        
        # Execute view creation
        print("📈 Creating views...")
        with db_manager.get_session() as session:
            session.execute(text(create_views_sql))
            session.commit()
        print("✅ Views created successfully")
        
        # Execute trigger creation
        print("⚡ Creating triggers...")
        with db_manager.get_session() as session:
            session.execute(text(create_triggers_sql))
            session.commit()
        print("✅ Triggers created successfully")
        
        # Insert sample data
        print("📝 Inserting sample baseline data...")
        with db_manager.get_session() as session:
            session.execute(text(insert_sample_data_sql))
            session.commit()
        print("✅ Sample data inserted successfully")
        
        print("\n🎉 Reporting system database setup completed successfully!")
        print("\nNext steps:")
        print("1. Run 'python scripts/demo_reporting.py' to test the system")
        print("2. Set up automated daily report generation")
        print("3. Configure alert notifications")
        
        return True
        
    except Exception as e:
        print(f"❌ Error setting up database: {e}")
        logger.error(f"Database setup failed: {e}")
        return False
    
    finally:
        db_manager.close()


if __name__ == "__main__":
    setup_reporting_tables()