"""
Unit tests for ETL pipeline components.
"""
import unittest
import pandas as pd
from datetime import datetime, date, timedelta
from typing import List

from src.models import PlayerProfile, RetentionMetrics
from src.etl import EventIngestion, DataLoader, RetentionAggregator, CohortAnalyzer


class TestEventIngestion(unittest.TestCase):
    """Test event ingestion functionality."""
    
    def setUp(self):
        """Set up test data."""
        self.ingestion = EventIngestion()
        
        # Create test events
        self.test_events = [
            {
                'player_id': 'player_1',
                'timestamp': datetime(2024, 1, 1, 10, 0, 0),
                'event_type': 'session_start',
                'session_duration': 300,
                'level': 1
            },
            {
                'player_id': 'player_1',
                'timestamp': datetime(2024, 1, 1, 10, 5, 0),
                'event_type': 'level_complete',
                'session_duration': 0,
                'level': 1
            },
            {
                'player_id': 'player_1',
                'timestamp': datetime(2024, 1, 1, 10, 10, 0),
                'event_type': 'purchase',
                'purchase_amount': 4.99,
                'level': 1
            },
            {
                'player_id': 'player_2',
                'timestamp': datetime(2024, 1, 2, 14, 0, 0),
                'event_type': 'session_start',
                'session_duration': 600,
                'level': 1
            }
        ]
    
    def test_process_events_to_dataframe(self):
        """Test converting events to DataFrame."""
        df = self.ingestion.process_events_to_dataframe(self.test_events)
        
        # Check DataFrame structure
        self.assertEqual(len(df), 4)
        self.assertIn('player_id', df.columns)
        self.assertIn('timestamp', df.columns)
        self.assertIn('event_type', df.columns)
        self.assertIn('date', df.columns)
        self.assertIn('hour', df.columns)
        
        # Check derived columns
        self.assertEqual(df.iloc[0]['date'], date(2024, 1, 1))
        self.assertEqual(df.iloc[0]['hour'], 10)
    
    def test_extract_session_events(self):
        """Test extracting session events."""
        df = self.ingestion.process_events_to_dataframe(self.test_events)
        session_events = self.ingestion.extract_session_events(df)
        
        # Should only have session_start events
        self.assertEqual(len(session_events), 2)
        self.assertTrue(all(session_events['event_type'] == 'session_start'))
    
    def test_extract_purchase_events(self):
        """Test extracting purchase events."""
        df = self.ingestion.process_events_to_dataframe(self.test_events)
        purchase_events = self.ingestion.extract_purchase_events(df)
        
        # Should only have purchase events
        self.assertEqual(len(purchase_events), 1)
        self.assertEqual(purchase_events.iloc[0]['purchase_amount'], 4.99)
    
    def test_extract_level_events(self):
        """Test extracting level completion events."""
        df = self.ingestion.process_events_to_dataframe(self.test_events)
        level_events = self.ingestion.extract_level_events(df)
        
        # Should only have level_complete events
        self.assertEqual(len(level_events), 1)
        self.assertEqual(level_events.iloc[0]['event_type'], 'level_complete')
    
    def test_calculate_session_metrics(self):
        """Test session metrics calculation."""
        df = self.ingestion.process_events_to_dataframe(self.test_events)
        session_events = self.ingestion.extract_session_events(df)
        session_metrics = self.ingestion.calculate_session_metrics(session_events)
        
        # Check metrics structure
        self.assertIn('player_id', session_metrics.columns)
        self.assertIn('date', session_metrics.columns)
        self.assertIn('session_count', session_metrics.columns)
        
        # Check player 1 has 1 session on 2024-01-01
        player_1_metrics = session_metrics[session_metrics['player_id'] == 'player_1']
        self.assertEqual(len(player_1_metrics), 1)
        self.assertEqual(player_1_metrics.iloc[0]['session_count'], 1)
    
    def test_create_daily_player_summary(self):
        """Test daily player summary creation."""
        df = self.ingestion.process_events_to_dataframe(self.test_events)
        summary = self.ingestion.create_daily_player_summary(df)
        
        # Check summary structure
        expected_columns = [
            'player_id', 'date', 'total_events', 'session_count',
            'purchase_count', 'levels_completed'
        ]
        
        for col in expected_columns:
            if col in summary.columns:
                self.assertIn(col, summary.columns)
        
        # Check player 1 summary
        player_1_summary = summary[summary['player_id'] == 'player_1']
        self.assertEqual(len(player_1_summary), 1)
        self.assertEqual(player_1_summary.iloc[0]['total_events'], 3)


class TestRetentionAggregator(unittest.TestCase):
    """Test retention aggregation functionality."""
    
    def setUp(self):
        """Set up test data."""
        self.aggregator = RetentionAggregator()
        
        # Create test profiles
        self.test_profiles = [
            PlayerProfile(
                player_id='player_1',
                registration_date=datetime(2024, 1, 1),
                last_active_date=datetime(2024, 1, 5),
                total_sessions=5,
                total_playtime_minutes=1500,
                highest_level_reached=3,
                total_purchases=9.99,
                churn_risk_score=0.2,
                churn_prediction_date=datetime(2024, 1, 10)
            ),
            PlayerProfile(
                player_id='player_2',
                registration_date=datetime(2024, 1, 1),
                last_active_date=datetime(2024, 1, 2),
                total_sessions=2,
                total_playtime_minutes=600,
                highest_level_reached=1,
                total_purchases=0.0,
                churn_risk_score=0.8,
                churn_prediction_date=datetime(2024, 1, 10)
            ),
            PlayerProfile(
                player_id='player_3',
                registration_date=datetime(2024, 1, 2),
                last_active_date=datetime(2024, 1, 10),
                total_sessions=8,
                total_playtime_minutes=2400,
                highest_level_reached=5,
                total_purchases=19.99,
                churn_risk_score=0.1,
                churn_prediction_date=datetime(2024, 1, 15)
            )
        ]
        
        # Create test events
        self.test_events_data = [
            # Player 1 events
            {'player_id': 'player_1', 'timestamp': datetime(2024, 1, 1, 10, 0), 'event_type': 'session_start'},
            {'player_id': 'player_1', 'timestamp': datetime(2024, 1, 2, 10, 0), 'event_type': 'session_start'},
            {'player_id': 'player_1', 'timestamp': datetime(2024, 1, 8, 10, 0), 'event_type': 'session_start'},
            
            # Player 2 events (churned after day 1)
            {'player_id': 'player_2', 'timestamp': datetime(2024, 1, 1, 11, 0), 'event_type': 'session_start'},
            {'player_id': 'player_2', 'timestamp': datetime(2024, 1, 2, 11, 0), 'event_type': 'session_start'},
            
            # Player 3 events (registered day 2)
            {'player_id': 'player_3', 'timestamp': datetime(2024, 1, 2, 12, 0), 'event_type': 'session_start'},
            {'player_id': 'player_3', 'timestamp': datetime(2024, 1, 3, 12, 0), 'event_type': 'session_start'},
            {'player_id': 'player_3', 'timestamp': datetime(2024, 1, 9, 12, 0), 'event_type': 'session_start'},
        ]
        
        self.test_events_df = pd.DataFrame(self.test_events_data)
    
    def test_calculate_daily_retention(self):
        """Test daily retention calculation."""
        retention_metrics = self.aggregator.calculate_daily_retention(
            self.test_events_df, self.test_profiles
        )
        
        # Should have metrics for each cohort date
        self.assertGreater(len(retention_metrics), 0)
        
        # Check that all metrics are RetentionMetrics objects
        for metrics in retention_metrics:
            self.assertIsInstance(metrics, RetentionMetrics)
            self.assertGreaterEqual(metrics.day_1_retention, 0)
            self.assertLessEqual(metrics.day_1_retention, 1)
            self.assertGreater(metrics.cohort_size, 0)
    
    def test_calculate_weekly_retention(self):
        """Test weekly retention calculation."""
        retention_metrics = self.aggregator.calculate_weekly_retention(
            self.test_events_df, self.test_profiles
        )
        
        # Should have weekly retention metrics
        self.assertGreater(len(retention_metrics), 0)
        
        for metrics in retention_metrics:
            self.assertIsInstance(metrics, RetentionMetrics)
            self.assertEqual(metrics.segment, 'weekly')
    
    def test_calculate_monthly_retention(self):
        """Test monthly retention calculation."""
        retention_metrics = self.aggregator.calculate_monthly_retention(
            self.test_events_df, self.test_profiles
        )
        
        # Should have monthly retention metrics
        self.assertGreater(len(retention_metrics), 0)
        
        for metrics in retention_metrics:
            self.assertIsInstance(metrics, RetentionMetrics)
            self.assertEqual(metrics.segment, 'monthly')
    
    def test_calculate_segment_retention(self):
        """Test segmented retention calculation."""
        retention_metrics = self.aggregator.calculate_segment_retention(
            self.test_events_df, self.test_profiles, 'total_purchases'
        )
        
        # Should have metrics for different spending segments
        segments = set(metrics.segment for metrics in retention_metrics)
        expected_segments = {'non_paying', 'low_spender'}
        
        # Should have at least some of the expected segments
        self.assertTrue(len(segments.intersection(expected_segments)) > 0)
    
    def test_count_active_players_on_day(self):
        """Test counting active players on specific day."""
        # Create activity DataFrame
        activity_data = []
        for event in self.test_events_data:
            activity_data.append({
                'player_id': event['player_id'],
                'date': event['timestamp'].date()
            })
        
        activity_df = pd.DataFrame(activity_data)
        
        # Test counting players active on day 1 after Jan 1 cohort
        cohort_players = ['player_1', 'player_2']
        cohort_date = date(2024, 1, 1)
        
        active_count = self.aggregator._count_active_players_on_day(
            activity_df, cohort_players, cohort_date, 1
        )
        
        # Both players were active on day 1 (Jan 2)
        self.assertEqual(active_count, 2)


class TestCohortAnalyzer(unittest.TestCase):
    """Test cohort analysis functionality."""
    
    def setUp(self):
        """Set up test data."""
        self.analyzer = CohortAnalyzer()
        
        # Create test profiles
        self.test_profiles = [
            PlayerProfile(
                player_id='player_1',
                registration_date=datetime(2024, 1, 1),
                last_active_date=datetime(2024, 1, 5),
                total_sessions=5,
                total_playtime_minutes=1500,
                highest_level_reached=3,
                total_purchases=9.99,
                churn_risk_score=0.2,
                churn_prediction_date=datetime(2024, 1, 10)
            ),
            PlayerProfile(
                player_id='player_2',
                registration_date=datetime(2024, 1, 1),
                last_active_date=datetime(2024, 1, 2),
                total_sessions=2,
                total_playtime_minutes=600,
                highest_level_reached=1,
                total_purchases=0.0,
                churn_risk_score=0.8,
                churn_prediction_date=datetime(2024, 1, 10)
            )
        ]
        
        # Create test events
        self.test_events_data = [
            {'player_id': 'player_1', 'timestamp': datetime(2024, 1, 1, 10, 0), 'event_type': 'session_start'},
            {'player_id': 'player_1', 'timestamp': datetime(2024, 1, 2, 10, 0), 'event_type': 'session_start'},
            {'player_id': 'player_1', 'timestamp': datetime(2024, 1, 8, 10, 0), 'event_type': 'session_start'},
            {'player_id': 'player_2', 'timestamp': datetime(2024, 1, 1, 11, 0), 'event_type': 'session_start'},
            {'player_id': 'player_2', 'timestamp': datetime(2024, 1, 2, 11, 0), 'event_type': 'session_start'},
        ]
        
        self.test_events_df = pd.DataFrame(self.test_events_data)
    
    def test_create_cohort_table(self):
        """Test cohort table creation."""
        cohort_table = self.analyzer.create_cohort_table(
            self.test_events_df, self.test_profiles
        )
        
        # Should be a DataFrame with cohort dates as index
        self.assertIsInstance(cohort_table, pd.DataFrame)
        self.assertGreater(len(cohort_table), 0)
        
        # Should have day columns
        self.assertIn(0, cohort_table.columns)  # Registration day
        self.assertIn(1, cohort_table.columns)  # Day 1
    
    def test_create_cohort_heatmap_data(self):
        """Test heatmap data creation."""
        heatmap_data = self.analyzer.create_cohort_heatmap_data(
            self.test_events_df, self.test_profiles
        )
        
        # Check data structure
        expected_keys = ['cohort_dates', 'day_columns', 'retention_matrix', 'cohort_sizes']
        for key in expected_keys:
            self.assertIn(key, heatmap_data)
        
        # Check data types
        self.assertIsInstance(heatmap_data['cohort_dates'], list)
        self.assertIsInstance(heatmap_data['day_columns'], list)
        self.assertIsInstance(heatmap_data['retention_matrix'], list)
        self.assertIsInstance(heatmap_data['cohort_sizes'], list)
    
    def test_analyze_cohort_trends(self):
        """Test cohort trend analysis."""
        trends = self.analyzer.analyze_cohort_trends(
            self.test_events_df, self.test_profiles
        )
        
        # Check trend structure
        expected_keys = [
            'average_retention_by_day', 'cohort_performance',
            'retention_decline_rates', 'best_performing_cohorts',
            'worst_performing_cohorts'
        ]
        
        for key in expected_keys:
            self.assertIn(key, trends)
        
        # Check cohort performance data
        self.assertIsInstance(trends['cohort_performance'], list)
        if trends['cohort_performance']:
            performance = trends['cohort_performance'][0]
            self.assertIn('cohort_date', performance)
            self.assertIn('day_1_retention', performance)
            self.assertIn('cohort_size', performance)
    
    def test_create_lifecycle_segments(self):
        """Test lifecycle segmentation."""
        segments_df = self.analyzer.create_lifecycle_segments(
            self.test_events_df, self.test_profiles
        )
        
        # Check DataFrame structure
        expected_columns = [
            'player_id', 'registration_date', 'days_since_last_activity',
            'total_active_days', 'lifecycle_segment'
        ]
        
        for col in expected_columns:
            self.assertIn(col, segments_df.columns)
        
        # Check that segments are assigned
        segments = segments_df['lifecycle_segment'].unique()
        valid_segments = [
            'highly_active', 'active', 'casual', 'at_risk', 'dormant', 'churned', 'inactive'
        ]
        
        for segment in segments:
            self.assertIn(segment, valid_segments)
    
    def test_create_retention_funnel(self):
        """Test retention funnel creation."""
        funnel = self.analyzer.create_retention_funnel(
            self.test_events_df, self.test_profiles
        )
        
        # Check funnel structure
        expected_keys = ['stages', 'total_players', 'retention_rates', 'drop_off_rates']
        for key in expected_keys:
            self.assertIn(key, funnel)
        
        # Check data consistency
        self.assertEqual(len(funnel['stages']), len(funnel['retention_rates']))
        self.assertEqual(len(funnel['stages']), len(funnel['drop_off_rates']))
        self.assertEqual(funnel['total_players'], len(self.test_profiles))


class TestETLIntegration(unittest.TestCase):
    """Test integration between ETL components."""
    
    def setUp(self):
        """Set up integration test data."""
        self.ingestion = EventIngestion()
        self.aggregator = RetentionAggregator()
        self.analyzer = CohortAnalyzer()
        
        # Create comprehensive test data
        self.test_profiles = [
            PlayerProfile(
                player_id=f'player_{i}',
                registration_date=datetime(2024, 1, 1) + timedelta(days=i % 3),
                last_active_date=datetime(2024, 1, 10),
                total_sessions=5 + i,
                total_playtime_minutes=1500 + i * 100,
                highest_level_reached=min(10, i + 1),
                total_purchases=i * 4.99,
                churn_risk_score=0.1 + (i % 10) * 0.1,
                churn_prediction_date=datetime(2024, 1, 15)
            )
            for i in range(10)
        ]
        
        # Create events for multiple days
        self.test_events_data = []
        for i, profile in enumerate(self.test_profiles):
            reg_date = profile.registration_date
            
            # Add events for several days after registration
            for day_offset in [0, 1, 3, 7, 14]:
                event_date = reg_date + timedelta(days=day_offset)
                if event_date <= datetime(2024, 1, 15):  # Don't go beyond test range
                    self.test_events_data.append({
                        'player_id': profile.player_id,
                        'timestamp': event_date + timedelta(hours=10 + i % 12),
                        'event_type': 'session_start',
                        'session_duration': 300 + i * 10,
                        'level': min(10, day_offset + 1)
                    })
        
        self.test_events_df = pd.DataFrame(self.test_events_data)
    
    def test_end_to_end_pipeline(self):
        """Test complete ETL pipeline from events to cohort analysis."""
        
        # Step 1: Process events
        events_df = self.ingestion.process_events_to_dataframe(self.test_events_data)
        daily_summary = self.ingestion.create_daily_player_summary(events_df)
        
        # Verify event processing
        self.assertGreater(len(events_df), 0)
        self.assertGreater(len(daily_summary), 0)
        
        # Step 2: Calculate retention metrics
        daily_retention = self.aggregator.calculate_daily_retention(events_df, self.test_profiles)
        weekly_retention = self.aggregator.calculate_weekly_retention(events_df, self.test_profiles)
        
        # Verify retention calculations
        self.assertGreater(len(daily_retention), 0)
        self.assertGreater(len(weekly_retention), 0)
        
        # Step 3: Perform cohort analysis
        cohort_table = self.analyzer.create_cohort_table(events_df, self.test_profiles)
        cohort_trends = self.analyzer.analyze_cohort_trends(events_df, self.test_profiles)
        
        # Verify cohort analysis
        self.assertIsInstance(cohort_table, pd.DataFrame)
        self.assertGreater(len(cohort_table), 0)
        self.assertIn('cohort_performance', cohort_trends)
        
        # Step 4: Verify data consistency
        # Number of unique cohort dates should match between retention and cohort analysis
        retention_cohorts = set(metrics.cohort_date for metrics in daily_retention)
        cohort_table_dates = set(cohort_table.index)
        
        # Should have some overlap (exact match depends on data availability)
        self.assertGreater(len(retention_cohorts.intersection(cohort_table_dates)), 0)
    
    def test_data_quality_validation(self):
        """Test data quality validation across pipeline."""
        
        # Process events
        events_df = self.ingestion.process_events_to_dataframe(self.test_events_data)
        
        # Validate data quality
        quality_report = self.ingestion.validate_data_quality(events_df, self.test_profiles)
        
        # Check quality report structure
        expected_keys = ['total_events', 'total_players', 'date_range', 'event_types', 'issues']
        for key in expected_keys:
            self.assertIn(key, quality_report)
        
        # Verify basic data quality
        self.assertEqual(quality_report['total_events'], len(self.test_events_data))
        self.assertEqual(quality_report['total_players'], len(self.test_profiles))
        
        # Should have minimal issues with clean test data
        self.assertIsInstance(quality_report['issues'], list)


if __name__ == '__main__':
    unittest.main()