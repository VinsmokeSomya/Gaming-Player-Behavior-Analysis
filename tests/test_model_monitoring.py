"""
Unit tests for model performance monitoring and retraining system.
"""
import pytest
import numpy as np
import pandas as pd
from datetime import date, timedelta
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os

from src.services.model_monitoring import (
    ModelPerformanceMonitor, 
    FeatureDriftDetector, 
    AutoRetrainingSystem
)
from src.models.churn_features import ChurnFeatures


class TestModelPerformanceMonitor:
    """Test cases for ModelPerformanceMonitor."""
    
    @pytest.fixture
    def sample_features(self):
        """Create sample churn features for testing."""
        features = []
        for i in range(100):
            days_since = i % 10
            # Ensure business logic: if days_since > 7, then sessions = 0
            sessions = max(0, 7 - (i % 8)) if days_since <= 7 else 0
            
            features.append(ChurnFeatures(
                player_id=f"player_{i}",
                days_since_last_session=days_since,
                sessions_last_7_days=sessions,
                avg_session_duration_minutes=30 + (i % 20),
                levels_completed_last_week=i % 15,
                purchases_last_30_days=float(i % 50),
                social_connections=i % 25,
                feature_date=date.today()
            ))
        return features
    
    @pytest.fixture
    def sample_labels(self):
        """Create sample labels for testing."""
        return [i % 2 == 0 for i in range(100)]  # Alternating True/False
    
    @pytest.fixture
    def monitor_with_mock_model(self):
        """Create monitor with mocked model."""
        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as tmp:
            model_path = tmp.name
        
        # Create monitor without loading existing model
        monitor = ModelPerformanceMonitor.__new__(ModelPerformanceMonitor)
        monitor.model_path = model_path
        monitor.predictor = Mock()
        monitor.accuracy_threshold = 0.75
        monitor.performance_history = []
        
        # Mock the predictor methods
        monitor.predictor.predict_churn_probability.return_value = [0.3, 0.7, 0.2, 0.8] * 25
        
        yield monitor
        
        # Cleanup
        if os.path.exists(model_path):
            os.unlink(model_path)
    
    def test_evaluate_daily_accuracy_success(self, monitor_with_mock_model, sample_features, sample_labels):
        """Test successful daily accuracy evaluation."""
        monitor = monitor_with_mock_model
        
        results = monitor.evaluate_daily_accuracy(sample_features, sample_labels)
        
        assert 'accuracy' in results
        assert 'precision' in results
        assert 'recall' in results
        assert 'f1_score' in results
        assert 'roc_auc' in results
        assert 'evaluation_date' in results
        assert 'holdout_size' in results
        assert 'meets_threshold' in results
        assert results['holdout_size'] == 100
        
        # Verify predictor was called
        monitor.predictor.predict_churn_probability.assert_called_once_with(sample_features)
    
    def test_evaluate_daily_accuracy_empty_data(self, monitor_with_mock_model):
        """Test evaluation with empty data raises error."""
        monitor = monitor_with_mock_model
        
        with pytest.raises(ValueError, match="Holdout dataset cannot be empty"):
            monitor.evaluate_daily_accuracy([], [])
    
    def test_evaluate_daily_accuracy_mismatched_lengths(self, monitor_with_mock_model, sample_features):
        """Test evaluation with mismatched feature/label lengths."""
        monitor = monitor_with_mock_model
        
        with pytest.raises(ValueError, match="Features and labels must have same length"):
            monitor.evaluate_daily_accuracy(sample_features, [True, False])  # Wrong length
    
    def test_check_accuracy_threshold_meets(self, monitor_with_mock_model):
        """Test accuracy threshold check when threshold is met."""
        monitor = monitor_with_mock_model
        monitor.accuracy_threshold = 0.8
        
        results = {'accuracy': 0.85}
        assert monitor.check_accuracy_threshold(results) is True
    
    def test_check_accuracy_threshold_fails(self, monitor_with_mock_model):
        """Test accuracy threshold check when threshold is not met."""
        monitor = monitor_with_mock_model
        monitor.accuracy_threshold = 0.8
        
        results = {'accuracy': 0.75}
        assert monitor.check_accuracy_threshold(results) is False
    
    def test_get_performance_trend_insufficient_data(self, monitor_with_mock_model):
        """Test performance trend with insufficient data."""
        monitor = monitor_with_mock_model
        
        trend = monitor.get_performance_trend()
        
        assert trend['trend'] == 'insufficient_data'
        assert trend['recent_accuracy'] is None
        assert trend['accuracy_change'] is None
        assert trend['days_analyzed'] == 0
    
    def test_get_performance_trend_improving(self, monitor_with_mock_model):
        """Test performance trend detection for improving accuracy."""
        monitor = monitor_with_mock_model
        
        # Add performance history with improving trend
        monitor.performance_history = [
            {'accuracy': 0.70},
            {'accuracy': 0.75},
            {'accuracy': 0.80},
            {'accuracy': 0.85}
        ]
        
        trend = monitor.get_performance_trend()
        
        assert trend['trend'] == 'improving'
        assert trend['recent_accuracy'] == 0.85
        assert abs(trend['accuracy_change'] - 0.15) < 0.001
        assert trend['days_analyzed'] == 4
        assert trend['slope'] > 0
    
    def test_get_performance_trend_declining(self, monitor_with_mock_model):
        """Test performance trend detection for declining accuracy."""
        monitor = monitor_with_mock_model
        
        # Add performance history with declining trend
        monitor.performance_history = [
            {'accuracy': 0.85},
            {'accuracy': 0.80},
            {'accuracy': 0.75},
            {'accuracy': 0.70}
        ]
        
        trend = monitor.get_performance_trend()
        
        assert trend['trend'] == 'declining'
        assert trend['recent_accuracy'] == 0.70
        assert abs(trend['accuracy_change'] - (-0.15)) < 0.001
        assert trend['days_analyzed'] == 4
        assert trend['slope'] < 0


class TestFeatureDriftDetector:
    """Test cases for FeatureDriftDetector."""
    
    @pytest.fixture
    def baseline_features(self):
        """Create baseline features for testing."""
        np.random.seed(42)  # For reproducible tests
        features = []
        for i in range(100):
            days_since = max(0, int(np.random.normal(5, 2)))
            # Ensure business logic: if days_since > 7, then sessions = 0
            sessions = int(np.random.poisson(3)) if days_since <= 7 else 0
            
            features.append(ChurnFeatures(
                player_id=f"player_{i}",
                days_since_last_session=days_since,
                sessions_last_7_days=sessions,
                avg_session_duration_minutes=max(0.0, np.random.normal(30, 10)),
                levels_completed_last_week=int(np.random.poisson(5)),
                purchases_last_30_days=max(0.0, np.random.exponential(10)),
                social_connections=int(np.random.poisson(8)),
                feature_date=date.today()
            ))
        return features
    
    @pytest.fixture
    def drifted_features(self):
        """Create features with distribution drift."""
        np.random.seed(123)  # Different seed for drift
        features = []
        for i in range(100):
            days_since = max(0, int(np.random.normal(8, 3)))  # Shifted mean
            # Ensure business logic: if days_since > 7, then sessions = 0
            sessions = int(np.random.poisson(2)) if days_since <= 7 else 0
            
            features.append(ChurnFeatures(
                player_id=f"player_{i}",
                days_since_last_session=days_since,
                sessions_last_7_days=sessions,
                avg_session_duration_minutes=max(0.0, np.random.normal(25, 15)),  # Different variance
                levels_completed_last_week=int(np.random.poisson(3)),  # Lower levels
                purchases_last_30_days=max(0.0, np.random.exponential(5)),  # Lower purchases
                social_connections=int(np.random.poisson(12)),  # Higher connections
                feature_date=date.today()
            ))
        return features
    
    @pytest.fixture
    def detector(self):
        """Create drift detector instance."""
        return FeatureDriftDetector()
    
    def test_set_baseline_distribution_success(self, detector, baseline_features):
        """Test successful baseline distribution setting."""
        detector.set_baseline_distribution(baseline_features)
        
        assert detector.baseline_distributions is not None
        assert 'baseline_date' in detector.baseline_distributions
        assert 'sample_size' in detector.baseline_distributions
        assert 'features' in detector.baseline_distributions
        assert detector.baseline_distributions['sample_size'] == 100
        
        # Check that all expected features are present
        expected_features = [
            'days_since_last_session', 'sessions_last_7_days',
            'avg_session_duration_minutes', 'levels_completed_last_week',
            'purchases_last_30_days', 'social_connections'
        ]
        
        for feature in expected_features:
            assert feature in detector.baseline_distributions['features']
            feature_stats = detector.baseline_distributions['features'][feature]
            assert 'mean' in feature_stats
            assert 'std' in feature_stats
            assert 'values' in feature_stats
    
    def test_set_baseline_distribution_empty_data(self, detector):
        """Test baseline setting with empty data raises error."""
        with pytest.raises(ValueError, match="Baseline features cannot be empty"):
            detector.set_baseline_distribution([])
    
    def test_detect_drift_no_baseline(self, detector, baseline_features):
        """Test drift detection without baseline raises error."""
        with pytest.raises(ValueError, match="Baseline distribution not set"):
            detector.detect_drift(baseline_features)
    
    def test_detect_drift_empty_current_data(self, detector, baseline_features):
        """Test drift detection with empty current data raises error."""
        detector.set_baseline_distribution(baseline_features)
        
        with pytest.raises(ValueError, match="Current features cannot be empty"):
            detector.detect_drift([])
    
    def test_detect_drift_no_drift(self, detector, baseline_features):
        """Test drift detection when no drift is present."""
        detector.set_baseline_distribution(baseline_features)
        
        # Use same distribution for current features (no drift)
        current_features = baseline_features[:50]  # Subset of same distribution
        
        results = detector.detect_drift(current_features)
        
        assert 'detection_date' in results
        assert 'baseline_date' in results
        assert 'overall_drift_detected' in results
        assert 'drifted_features' in results
        assert 'features' in results
        
        # Should not detect drift for same distribution
        assert results['overall_drift_detected'] is False
        assert len(results['drifted_features']) == 0
    
    def test_detect_drift_with_drift(self, detector, baseline_features, drifted_features):
        """Test drift detection when drift is present."""
        detector.set_baseline_distribution(baseline_features)
        
        results = detector.detect_drift(drifted_features)
        
        assert results['overall_drift_detected'] is True
        assert len(results['drifted_features']) > 0
        
        # Check individual feature results
        for feature_name, feature_result in results['features'].items():
            assert 'drift_detected' in feature_result
            assert 'ks_statistic' in feature_result
            assert 'ks_p_value' in feature_result
            assert 'mw_statistic' in feature_result
            assert 'mw_p_value' in feature_result
            assert 'baseline_mean' in feature_result
            assert 'current_mean' in feature_result
            assert 'mean_shift_percent' in feature_result


class TestAutoRetrainingSystem:
    """Test cases for AutoRetrainingSystem."""
    
    @pytest.fixture
    def sample_features(self):
        """Create sample features for testing."""
        features = []
        for i in range(50):
            days_since = i % 10
            # Ensure business logic: if days_since > 7, then sessions = 0
            sessions = max(0, 7 - (i % 8)) if days_since <= 7 else 0
            
            features.append(ChurnFeatures(
                player_id=f"player_{i}",
                days_since_last_session=days_since,
                sessions_last_7_days=sessions,
                avg_session_duration_minutes=30 + (i % 20),
                levels_completed_last_week=i % 15,
                purchases_last_30_days=float(i % 50),
                social_connections=i % 25,
                feature_date=date.today()
            ))
        return features
    
    @pytest.fixture
    def sample_labels(self):
        """Create sample labels for testing."""
        return [i % 2 == 0 for i in range(50)]
    
    @pytest.fixture
    def retraining_system(self):
        """Create auto-retraining system with mocked components."""
        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as tmp:
            model_path = tmp.name
        
        # Create system without loading existing model
        system = AutoRetrainingSystem.__new__(AutoRetrainingSystem)
        system.model_path = model_path
        system.retraining_history = []
        
        # Mock the performance monitor
        system.performance_monitor = Mock()
        system.performance_monitor.accuracy_threshold = 0.75
        system.performance_monitor.get_performance_trend.return_value = {
            'trend': 'stable',
            'accuracy_change': 0.0
        }
        
        # Mock the predictor
        system.performance_monitor.predictor = Mock()
        system.performance_monitor.predictor.prepare_training_data.return_value = (
            np.random.rand(50, 10), np.random.randint(0, 2, 50)
        )
        system.performance_monitor.predictor.train_models.return_value = {
            'random_forest': {'metrics': {'accuracy': 0.85}}
        }
        system.performance_monitor.predictor.best_model_name = 'random_forest'
        system.performance_monitor.predictor.save_model = Mock()
        
        # Mock the drift detector
        system.drift_detector = Mock()
        
        yield system
        
        # Cleanup
        if os.path.exists(model_path):
            os.unlink(model_path)
    
    def test_should_retrain_model_accuracy_below_threshold(self, retraining_system):
        """Test retraining decision when accuracy is below threshold."""
        performance_results = {
            'accuracy': 0.70,
            'meets_threshold': False
        }
        
        should_retrain, reasons = retraining_system.should_retrain_model(performance_results)
        
        assert should_retrain is True
        assert len(reasons) == 1
        assert "below threshold" in reasons[0]
    
    def test_should_retrain_model_declining_trend(self, retraining_system):
        """Test retraining decision when accuracy trend is declining."""
        performance_results = {
            'accuracy': 0.80,
            'meets_threshold': True
        }
        
        # Mock declining trend
        retraining_system.performance_monitor.get_performance_trend.return_value = {
            'trend': 'declining',
            'accuracy_change': -0.08
        }
        
        should_retrain, reasons = retraining_system.should_retrain_model(performance_results)
        
        assert should_retrain is True
        assert len(reasons) == 1
        assert "Declining accuracy trend" in reasons[0]
    
    def test_should_retrain_model_feature_drift(self, retraining_system):
        """Test retraining decision when feature drift is detected."""
        performance_results = {
            'accuracy': 0.80,
            'meets_threshold': True
        }
        
        drift_results = {
            'overall_drift_detected': True,
            'drifted_features': ['days_since_last_session', 'sessions_last_7_days']
        }
        
        should_retrain, reasons = retraining_system.should_retrain_model(
            performance_results, drift_results
        )
        
        assert should_retrain is True
        assert len(reasons) == 1
        assert "Feature drift detected" in reasons[0]
    
    def test_should_retrain_model_no_issues(self, retraining_system):
        """Test retraining decision when no issues are detected."""
        performance_results = {
            'accuracy': 0.85,
            'meets_threshold': True
        }
        
        drift_results = {
            'overall_drift_detected': False,
            'drifted_features': []
        }
        
        should_retrain, reasons = retraining_system.should_retrain_model(
            performance_results, drift_results
        )
        
        assert should_retrain is False
        assert len(reasons) == 0
    
    def test_trigger_retraining_success(self, retraining_system, sample_features, sample_labels):
        """Test successful model retraining."""
        reasons = ["Accuracy below threshold"]
        
        results = retraining_system.trigger_retraining(
            sample_features, sample_labels, reasons
        )
        
        assert 'retrain_date' in results
        assert 'reasons' in results
        assert 'training_size' in results
        assert 'best_model' in results
        assert 'training_results' in results
        
        assert results['reasons'] == reasons
        assert results['training_size'] == 50
        assert results['best_model'] == 'random_forest'
        
        # Verify methods were called
        retraining_system.performance_monitor.predictor.prepare_training_data.assert_called_once()
        retraining_system.performance_monitor.predictor.train_models.assert_called_once()
        retraining_system.performance_monitor.predictor.save_model.assert_called_once()
    
    def test_trigger_retraining_empty_data(self, retraining_system):
        """Test retraining with empty data raises error."""
        with pytest.raises(ValueError, match="Training data cannot be empty"):
            retraining_system.trigger_retraining([], [], ["Test reason"])
    
    @patch('src.services.model_monitoring.logger')
    def test_run_monitoring_cycle_complete(self, mock_logger, retraining_system, sample_features, sample_labels):
        """Test complete monitoring cycle execution."""
        # Mock performance evaluation
        performance_results = {
            'accuracy': 0.70,
            'meets_threshold': False
        }
        retraining_system.performance_monitor.evaluate_daily_accuracy.return_value = performance_results
        
        # Mock drift detection
        drift_results = {
            'overall_drift_detected': True,
            'drifted_features': ['days_since_last_session']
        }
        retraining_system.drift_detector.baseline_distributions = {'test': 'data'}
        retraining_system.drift_detector.detect_drift = Mock(return_value=drift_results)
        
        # Run monitoring cycle
        results = retraining_system.run_monitoring_cycle(
            holdout_features=sample_features,
            holdout_labels=sample_labels,
            current_features=sample_features,
            training_features=sample_features,
            training_labels=sample_labels
        )
        
        assert 'cycle_date' in results
        assert 'performance_evaluation' in results
        assert 'drift_detection' in results
        assert 'retraining_triggered' in results
        assert 'retraining_results' in results
        
        assert results['performance_evaluation'] == performance_results
        assert results['drift_detection'] == drift_results
        assert results['retraining_triggered'] is True
        
        # Verify methods were called
        retraining_system.performance_monitor.evaluate_daily_accuracy.assert_called_once()
        retraining_system.drift_detector.detect_drift.assert_called_once()
    
    def test_run_monitoring_cycle_no_retraining_data(self, retraining_system, sample_features, sample_labels):
        """Test monitoring cycle when retraining is needed but no training data provided."""
        # Mock performance evaluation that triggers retraining
        performance_results = {
            'accuracy': 0.70,
            'meets_threshold': False
        }
        retraining_system.performance_monitor.evaluate_daily_accuracy.return_value = performance_results
        
        # Mock drift detector to avoid baseline setting
        retraining_system.drift_detector.baseline_distributions = {'test': 'data'}
        retraining_system.drift_detector.detect_drift.return_value = {
            'overall_drift_detected': False,
            'drifted_features': []
        }
        
        # Run monitoring cycle without training data
        results = retraining_system.run_monitoring_cycle(
            holdout_features=sample_features,
            holdout_labels=sample_labels,
            current_features=sample_features
            # No training_features or training_labels provided
        )
        
        assert results['retraining_triggered'] is False
        assert results['retraining_results'] is not None
        assert 'error' in results['retraining_results']
        assert 'No training data provided' in results['retraining_results']['error']


if __name__ == '__main__':
    pytest.main([__file__])