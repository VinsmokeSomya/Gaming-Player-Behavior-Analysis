#!/usr/bin/env python3
"""
Demo script for model performance monitoring and retraining system.

This script demonstrates how to use the model monitoring system to:
1. Evaluate model performance against holdout data
2. Detect feature drift in production data
3. Automatically trigger model retraining when needed
"""
import sys
import os
from datetime import date, timedelta
import numpy as np

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.churn_features import ChurnFeatures


def create_sample_features(num_samples: int, seed: int = 42, drift: bool = False) -> list:
    """Create sample churn features for demonstration."""
    np.random.seed(seed)
    features = []
    
    for i in range(num_samples):
        if drift:
            # Create drifted distribution
            days_since = max(0, int(np.random.normal(8, 3)))  # Higher mean
            sessions = int(np.random.poisson(1)) if days_since <= 7 else 0  # Lower sessions
            duration = max(0.0, np.random.normal(20, 15))  # Lower duration
            levels = int(np.random.poisson(2))  # Lower levels
            purchases = max(0.0, np.random.exponential(3))  # Lower purchases
            connections = int(np.random.poisson(15))  # Higher connections
        else:
            # Create normal distribution
            days_since = max(0, int(np.random.normal(4, 2)))
            sessions = int(np.random.poisson(4)) if days_since <= 7 else 0
            duration = max(0.0, np.random.normal(35, 10))
            levels = int(np.random.poisson(6))
            purchases = max(0.0, np.random.exponential(12))
            connections = int(np.random.poisson(8))
        
        features.append(ChurnFeatures(
            player_id=f"player_{i}",
            days_since_last_session=days_since,
            sessions_last_7_days=sessions,
            avg_session_duration_minutes=duration,
            levels_completed_last_week=levels,
            purchases_last_30_days=purchases,
            social_connections=connections,
            feature_date=date.today()
        ))
    
    return features


def create_sample_labels(num_samples: int, churn_rate: float = 0.3) -> list:
    """Create sample churn labels."""
    np.random.seed(42)
    return [np.random.random() < churn_rate for _ in range(num_samples)]


def demo_performance_monitoring():
    """Demonstrate model performance monitoring."""
    print("=" * 60)
    print("MODEL PERFORMANCE MONITORING DEMO")
    print("=" * 60)
    
    # Import here to avoid relative import issues
    from services.model_monitoring import ModelPerformanceMonitor
    
    # Create sample data
    holdout_features = create_sample_features(200)
    holdout_labels = create_sample_labels(200)
    
    print(f"Created holdout dataset with {len(holdout_features)} samples")
    
    # Initialize monitor (without loading existing model)
    monitor = ModelPerformanceMonitor.__new__(ModelPerformanceMonitor)
    monitor.model_path = "models/churn_predictor.joblib"
    monitor.accuracy_threshold = 0.75
    monitor.performance_history = []
    
    # Mock predictor for demo
    class MockPredictor:
        def predict_churn_probability(self, features):
            # Simulate predictions with some accuracy
            np.random.seed(123)
            return [0.3 + 0.4 * np.random.random() for _ in features]
    
    monitor.predictor = MockPredictor()
    
    # Evaluate performance
    print("\n1. Evaluating model performance...")
    results = monitor.evaluate_daily_accuracy(holdout_features, holdout_labels)
    
    print(f"   Accuracy: {results['accuracy']:.4f}")
    print(f"   Precision: {results['precision']:.4f}")
    print(f"   Recall: {results['recall']:.4f}")
    print(f"   F1 Score: {results['f1_score']:.4f}")
    print(f"   ROC AUC: {results['roc_auc']:.4f}")
    print(f"   Meets Threshold: {results['meets_threshold']}")
    
    # Simulate performance history
    print("\n2. Simulating performance trend...")
    monitor.performance_history = [
        {'accuracy': 0.82, 'evaluation_date': '2025-09-26'},
        {'accuracy': 0.80, 'evaluation_date': '2025-09-27'},
        {'accuracy': 0.78, 'evaluation_date': '2025-09-28'},
        {'accuracy': 0.76, 'evaluation_date': '2025-09-29'},
        {'accuracy': results['accuracy'], 'evaluation_date': '2025-09-30'}
    ]
    
    trend = monitor.get_performance_trend()
    print(f"   Trend: {trend['trend']}")
    print(f"   Recent Accuracy: {trend['recent_accuracy']:.4f}")
    print(f"   Accuracy Change: {trend['accuracy_change']:.4f}")
    print(f"   Days Analyzed: {trend['days_analyzed']}")


def demo_drift_detection():
    """Demonstrate feature drift detection."""
    print("\n" + "=" * 60)
    print("FEATURE DRIFT DETECTION DEMO")
    print("=" * 60)
    
    # Import here to avoid relative import issues
    from services.model_monitoring import FeatureDriftDetector
    
    # Create baseline and current features
    baseline_features = create_sample_features(500, seed=42, drift=False)
    current_features = create_sample_features(300, seed=123, drift=True)
    
    print(f"Created baseline dataset with {len(baseline_features)} samples")
    print(f"Created current dataset with {len(current_features)} samples")
    
    # Initialize drift detector
    detector = FeatureDriftDetector()
    
    # Set baseline
    print("\n1. Setting baseline distribution...")
    detector.set_baseline_distribution(baseline_features)
    print("   Baseline distribution set successfully")
    
    # Detect drift
    print("\n2. Detecting feature drift...")
    drift_results = detector.detect_drift(current_features)
    
    print(f"   Overall Drift Detected: {drift_results['overall_drift_detected']}")
    print(f"   Drifted Features: {drift_results['drifted_features']}")
    
    if drift_results['overall_drift_detected']:
        print("\n   Feature-level drift analysis:")
        for feature_name, feature_result in drift_results['features'].items():
            if feature_result['drift_detected']:
                print(f"   - {feature_name}:")
                print(f"     * KS p-value: {feature_result['ks_p_value']:.6f}")
                print(f"     * Mean shift: {feature_result['mean_shift_percent']:.2f}%")
                print(f"     * Baseline mean: {feature_result['baseline_mean']:.2f}")
                print(f"     * Current mean: {feature_result['current_mean']:.2f}")


def demo_auto_retraining():
    """Demonstrate automatic retraining system."""
    print("\n" + "=" * 60)
    print("AUTOMATIC RETRAINING SYSTEM DEMO")
    print("=" * 60)
    
    # Import here to avoid relative import issues
    from services.model_monitoring import AutoRetrainingSystem
    
    # Create datasets
    holdout_features = create_sample_features(200, seed=42)
    holdout_labels = create_sample_labels(200)
    current_features = create_sample_features(300, seed=123, drift=True)
    training_features = create_sample_features(1000, seed=456)
    training_labels = create_sample_labels(1000)
    
    print(f"Created datasets:")
    print(f"  - Holdout: {len(holdout_features)} samples")
    print(f"  - Current: {len(current_features)} samples")
    print(f"  - Training: {len(training_features)} samples")
    
    # Initialize auto-retraining system (without loading model)
    system = AutoRetrainingSystem.__new__(AutoRetrainingSystem)
    system.model_path = "models/churn_predictor.joblib"
    system.retraining_history = []
    
    # Mock components for demo
    class MockPerformanceMonitor:
        def __init__(self):
            self.accuracy_threshold = 0.75
        
        def evaluate_daily_accuracy(self, features, labels, date=None):
            return {
                'accuracy': 0.72,  # Below threshold
                'precision': 0.70,
                'recall': 0.68,
                'f1_score': 0.69,
                'roc_auc': 0.75,
                'meets_threshold': False,
                'evaluation_date': date.isoformat() if date else '2025-09-30'
            }
        
        def get_performance_trend(self):
            return {
                'trend': 'declining',
                'accuracy_change': -0.08,
                'recent_accuracy': 0.72
            }
    
    class MockDriftDetector:
        def __init__(self):
            self.baseline_distributions = {'test': 'data'}
        
        def detect_drift(self, features, date=None):
            return {
                'overall_drift_detected': True,
                'drifted_features': ['days_since_last_session', 'sessions_last_7_days'],
                'detection_date': date.isoformat() if date else '2025-09-30'
            }
    
    system.performance_monitor = MockPerformanceMonitor()
    system.drift_detector = MockDriftDetector()
    
    # Run monitoring cycle
    print("\n1. Running complete monitoring cycle...")
    
    # Mock the retraining process
    def mock_trigger_retraining(features, labels, reasons, date=None):
        return {
            'retrain_date': date.isoformat() if date else '2025-09-30',
            'reasons': reasons,
            'training_size': len(features),
            'best_model': 'random_forest',
            'training_results': {
                'random_forest': {'metrics': {'accuracy': 0.85, 'roc_auc': 0.88}}
            }
        }
    
    system.trigger_retraining = mock_trigger_retraining
    
    # Check if retraining should be triggered
    performance_results = system.performance_monitor.evaluate_daily_accuracy(
        holdout_features, holdout_labels
    )
    drift_results = system.drift_detector.detect_drift(current_features)
    
    should_retrain, reasons = system.should_retrain_model(performance_results, drift_results)
    
    print(f"   Should Retrain: {should_retrain}")
    print(f"   Reasons: {reasons}")
    
    if should_retrain:
        print("\n2. Triggering model retraining...")
        retraining_results = system.trigger_retraining(
            training_features, training_labels, reasons
        )
        
        print(f"   Retraining completed successfully")
        print(f"   Training Size: {retraining_results['training_size']}")
        print(f"   Best Model: {retraining_results['best_model']}")
        print(f"   New Accuracy: {retraining_results['training_results']['random_forest']['metrics']['accuracy']:.4f}")


def main():
    """Run all monitoring demos."""
    print("PLAYER RETENTION ANALYTICS - MODEL MONITORING DEMO")
    print("This demo shows the model performance monitoring and retraining system")
    
    try:
        demo_performance_monitoring()
        demo_drift_detection()
        demo_auto_retraining()
        
        print("\n" + "=" * 60)
        print("DEMO COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print("\nKey Features Demonstrated:")
        print("✓ Daily model accuracy evaluation")
        print("✓ Performance trend analysis")
        print("✓ Feature drift detection using statistical tests")
        print("✓ Automatic retraining triggers")
        print("✓ Complete monitoring cycle execution")
        
    except Exception as e:
        print(f"\nError running demo: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())