"""
Model Performance Monitoring and Retraining System.

This module implements comprehensive model monitoring including accuracy evaluation,
drift detection, and automatic retraining triggers for the churn prediction model.
"""
import logging
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from scipy import stats
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib
import os

from src.models.churn_features import ChurnFeatures
from src.config import app_config
from src.services.ml_pipeline import ChurnPredictor, ModelEvaluator

logger = logging.getLogger(__name__)


class ModelPerformanceMonitor:
    """Monitor model performance and trigger retraining when needed."""
    
    def __init__(self, model_path: str = "models/churn_predictor.joblib"):
        """
        Initialize model performance monitor.
        
        Args:
            model_path: Path to the trained model file
        """
        self.model_path = model_path
        self.predictor = ChurnPredictor()
        self.accuracy_threshold = app_config.model_retrain_threshold
        self.performance_history = []
        
        # Load existing model if available
        if os.path.exists(model_path):
            self.predictor.load_model(model_path)
    
    def evaluate_daily_accuracy(
        self, 
        holdout_features: List[ChurnFeatures], 
        holdout_labels: List[bool],
        evaluation_date: date = None
    ) -> Dict[str, Any]:
        """
        Evaluate model accuracy against holdout dataset.
        
        Args:
            holdout_features: Holdout dataset features
            holdout_labels: Holdout dataset labels
            evaluation_date: Date of evaluation (defaults to today)
            
        Returns:
            Dictionary with evaluation results
        """
        if evaluation_date is None:
            evaluation_date = date.today()
        
        if not holdout_features or not holdout_labels:
            raise ValueError("Holdout dataset cannot be empty")
        
        if len(holdout_features) != len(holdout_labels):
            raise ValueError("Features and labels must have same length")
        
        logger.info(f"Evaluating model accuracy for {evaluation_date}")
        
        try:
            # Get predictions
            y_pred_proba = self.predictor.predict_churn_probability(holdout_features)
            y_pred = [prob >= 0.5 for prob in y_pred_proba]
            y_true = holdout_labels
            
            # Calculate metrics
            evaluation_results = ModelEvaluator.evaluate_model_performance(
                np.array(y_true), 
                np.array(y_pred), 
                np.array(y_pred_proba)
            )
            
            # Add metadata
            evaluation_results.update({
                'evaluation_date': evaluation_date.isoformat(),
                'holdout_size': len(holdout_features),
                'model_path': self.model_path,
                'meets_threshold': evaluation_results['accuracy'] >= self.accuracy_threshold
            })
            
            # Store in performance history
            self.performance_history.append(evaluation_results)
            
            logger.info(
                f"Model accuracy: {evaluation_results['accuracy']:.4f} "
                f"(threshold: {self.accuracy_threshold})"
            )
            
            return evaluation_results
            
        except Exception as e:
            logger.error(f"Error evaluating model accuracy: {str(e)}")
            raise
    
    def check_accuracy_threshold(self, evaluation_results: Dict[str, Any]) -> bool:
        """
        Check if model accuracy meets the required threshold.
        
        Args:
            evaluation_results: Results from evaluate_daily_accuracy
            
        Returns:
            True if accuracy meets threshold, False otherwise
        """
        accuracy = evaluation_results.get('accuracy', 0.0)
        meets_threshold = accuracy >= self.accuracy_threshold
        
        if not meets_threshold:
            logger.warning(
                f"Model accuracy {accuracy:.4f} below threshold {self.accuracy_threshold}"
            )
        
        return meets_threshold
    
    def get_performance_trend(self, days: int = 7) -> Dict[str, Any]:
        """
        Analyze performance trend over recent days.
        
        Args:
            days: Number of recent days to analyze
            
        Returns:
            Dictionary with trend analysis
        """
        if len(self.performance_history) < 2:
            return {
                'trend': 'insufficient_data',
                'recent_accuracy': None,
                'accuracy_change': None,
                'days_analyzed': 0
            }
        
        # Get recent performance data
        recent_history = self.performance_history[-days:]
        accuracies = [result['accuracy'] for result in recent_history]
        
        # Calculate trend
        if len(accuracies) >= 2:
            # Linear regression to detect trend
            x = np.arange(len(accuracies))
            slope, _, _, _, _ = stats.linregress(x, accuracies)
            
            trend = 'improving' if slope > 0.01 else 'declining' if slope < -0.01 else 'stable'
            accuracy_change = accuracies[-1] - accuracies[0]
        else:
            trend = 'stable'
            accuracy_change = 0.0
        
        return {
            'trend': trend,
            'recent_accuracy': accuracies[-1] if accuracies else None,
            'accuracy_change': accuracy_change,
            'days_analyzed': len(accuracies),
            'slope': slope if len(accuracies) >= 2 else 0.0
        }


class FeatureDriftDetector:
    """Detect feature distribution drift in model inputs."""
    
    def __init__(self):
        """Initialize drift detector."""
        self.baseline_distributions = {}
        self.drift_threshold = 0.05  # p-value threshold for statistical tests
    
    def set_baseline_distribution(
        self, 
        baseline_features: List[ChurnFeatures],
        baseline_date: date = None
    ) -> None:
        """
        Set baseline feature distributions for drift detection.
        
        Args:
            baseline_features: Baseline dataset features
            baseline_date: Date of baseline (defaults to today)
        """
        if baseline_date is None:
            baseline_date = date.today()
        
        if not baseline_features:
            raise ValueError("Baseline features cannot be empty")
        
        logger.info(f"Setting baseline distribution with {len(baseline_features)} samples")
        
        # Convert to DataFrame for easier processing
        data = []
        for cf in baseline_features:
            data.append({
                'days_since_last_session': cf.days_since_last_session,
                'sessions_last_7_days': cf.sessions_last_7_days,
                'avg_session_duration_minutes': cf.avg_session_duration_minutes,
                'levels_completed_last_week': cf.levels_completed_last_week,
                'purchases_last_30_days': cf.purchases_last_30_days,
                'social_connections': cf.social_connections
            })
        
        df = pd.DataFrame(data)
        
        # Store baseline statistics for each feature
        self.baseline_distributions = {
            'baseline_date': baseline_date.isoformat(),
            'sample_size': len(baseline_features),
            'features': {}
        }
        
        for column in df.columns:
            self.baseline_distributions['features'][column] = {
                'mean': df[column].mean(),
                'std': df[column].std(),
                'median': df[column].median(),
                'q25': df[column].quantile(0.25),
                'q75': df[column].quantile(0.75),
                'min': df[column].min(),
                'max': df[column].max(),
                'values': df[column].values.tolist()  # Store for statistical tests
            }
    
    def detect_drift(
        self, 
        current_features: List[ChurnFeatures],
        detection_date: date = None
    ) -> Dict[str, Any]:
        """
        Detect feature drift compared to baseline distribution.
        
        Args:
            current_features: Current dataset features
            detection_date: Date of detection (defaults to today)
            
        Returns:
            Dictionary with drift detection results
        """
        if detection_date is None:
            detection_date = date.today()
        
        if not self.baseline_distributions:
            raise ValueError("Baseline distribution not set. Call set_baseline_distribution first.")
        
        if not current_features:
            raise ValueError("Current features cannot be empty")
        
        logger.info(f"Detecting feature drift for {detection_date}")
        
        # Convert current features to DataFrame
        data = []
        for cf in current_features:
            data.append({
                'days_since_last_session': cf.days_since_last_session,
                'sessions_last_7_days': cf.sessions_last_7_days,
                'avg_session_duration_minutes': cf.avg_session_duration_minutes,
                'levels_completed_last_week': cf.levels_completed_last_week,
                'purchases_last_30_days': cf.purchases_last_30_days,
                'social_connections': cf.social_connections
            })
        
        current_df = pd.DataFrame(data)
        
        # Perform drift detection for each feature
        drift_results = {
            'detection_date': detection_date.isoformat(),
            'baseline_date': self.baseline_distributions['baseline_date'],
            'current_sample_size': len(current_features),
            'baseline_sample_size': self.baseline_distributions['sample_size'],
            'features': {},
            'overall_drift_detected': False,
            'drifted_features': []
        }
        
        for feature_name in current_df.columns:
            baseline_values = self.baseline_distributions['features'][feature_name]['values']
            current_values = current_df[feature_name].values
            
            # Perform Kolmogorov-Smirnov test
            ks_statistic, ks_p_value = stats.ks_2samp(baseline_values, current_values)
            
            # Perform Mann-Whitney U test (non-parametric)
            mw_statistic, mw_p_value = stats.mannwhitneyu(
                baseline_values, current_values, alternative='two-sided'
            )
            
            # Calculate distribution shift metrics
            baseline_mean = self.baseline_distributions['features'][feature_name]['mean']
            current_mean = current_df[feature_name].mean()
            mean_shift = abs(current_mean - baseline_mean) / baseline_mean if baseline_mean != 0 else 0
            
            # Determine if drift detected
            drift_detected = (
                ks_p_value < self.drift_threshold or 
                mw_p_value < self.drift_threshold or
                mean_shift > 0.2  # 20% mean shift threshold
            )
            
            drift_results['features'][feature_name] = {
                'drift_detected': drift_detected,
                'ks_statistic': ks_statistic,
                'ks_p_value': ks_p_value,
                'mw_statistic': float(mw_statistic),
                'mw_p_value': mw_p_value,
                'baseline_mean': baseline_mean,
                'current_mean': current_mean,
                'mean_shift_percent': mean_shift * 100,
                'baseline_std': self.baseline_distributions['features'][feature_name]['std'],
                'current_std': current_df[feature_name].std()
            }
            
            if drift_detected:
                drift_results['drifted_features'].append(feature_name)
                drift_results['overall_drift_detected'] = True
        
        if drift_results['overall_drift_detected']:
            logger.warning(
                f"Feature drift detected in: {', '.join(drift_results['drifted_features'])}"
            )
        else:
            logger.info("No significant feature drift detected")
        
        return drift_results


class AutoRetrainingSystem:
    """Automatic model retraining system."""
    
    def __init__(self, model_path: str = "models/churn_predictor.joblib"):
        """
        Initialize auto-retraining system.
        
        Args:
            model_path: Path to the model file
        """
        self.model_path = model_path
        self.performance_monitor = ModelPerformanceMonitor(model_path)
        self.drift_detector = FeatureDriftDetector()
        self.retraining_history = []
    
    def should_retrain_model(
        self, 
        performance_results: Dict[str, Any],
        drift_results: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, List[str]]:
        """
        Determine if model should be retrained based on performance and drift.
        
        Args:
            performance_results: Results from performance evaluation
            drift_results: Results from drift detection (optional)
            
        Returns:
            Tuple of (should_retrain, reasons)
        """
        should_retrain = False
        reasons = []
        
        # Check accuracy threshold
        if not performance_results.get('meets_threshold', True):
            should_retrain = True
            accuracy = performance_results.get('accuracy', 0.0)
            reasons.append(f"Accuracy {accuracy:.4f} below threshold {self.performance_monitor.accuracy_threshold}")
        
        # Check performance trend
        trend_analysis = self.performance_monitor.get_performance_trend()
        if trend_analysis['trend'] == 'declining' and trend_analysis['accuracy_change'] < -0.05:
            should_retrain = True
            reasons.append(f"Declining accuracy trend: {trend_analysis['accuracy_change']:.4f}")
        
        # Check feature drift
        if drift_results and drift_results.get('overall_drift_detected', False):
            should_retrain = True
            drifted_features = drift_results.get('drifted_features', [])
            reasons.append(f"Feature drift detected in: {', '.join(drifted_features)}")
        
        return should_retrain, reasons
    
    def trigger_retraining(
        self, 
        training_features: List[ChurnFeatures],
        training_labels: List[bool],
        reasons: List[str],
        retrain_date: date = None
    ) -> Dict[str, Any]:
        """
        Trigger model retraining with new data.
        
        Args:
            training_features: New training dataset features
            training_labels: New training dataset labels
            reasons: Reasons for retraining
            retrain_date: Date of retraining (defaults to today)
            
        Returns:
            Dictionary with retraining results
        """
        if retrain_date is None:
            retrain_date = date.today()
        
        if not training_features or not training_labels:
            raise ValueError("Training data cannot be empty")
        
        logger.info(f"Starting model retraining on {retrain_date}")
        logger.info(f"Retraining reasons: {', '.join(reasons)}")
        
        try:
            # Prepare training data
            X, y = self.performance_monitor.predictor.prepare_training_data(
                training_features, training_labels
            )
            
            # Train new models
            training_results = self.performance_monitor.predictor.train_models(X, y)
            
            # Save new model
            backup_path = f"{self.model_path}.backup.{retrain_date.isoformat()}"
            if os.path.exists(self.model_path):
                # Backup old model
                os.rename(self.model_path, backup_path)
                logger.info(f"Old model backed up to {backup_path}")
            
            self.performance_monitor.predictor.save_model(self.model_path)
            
            # Record retraining event
            retraining_record = {
                'retrain_date': retrain_date.isoformat(),
                'reasons': reasons,
                'training_size': len(training_features),
                'best_model': self.performance_monitor.predictor.best_model_name,
                'training_results': training_results,
                'backup_path': backup_path if os.path.exists(backup_path) else None
            }
            
            self.retraining_history.append(retraining_record)
            
            logger.info(
                f"Model retraining completed. New best model: "
                f"{self.performance_monitor.predictor.best_model_name}"
            )
            
            return retraining_record
            
        except Exception as e:
            logger.error(f"Error during model retraining: {str(e)}")
            raise
    
    def run_monitoring_cycle(
        self,
        holdout_features: List[ChurnFeatures],
        holdout_labels: List[bool],
        current_features: List[ChurnFeatures],
        training_features: Optional[List[ChurnFeatures]] = None,
        training_labels: Optional[List[bool]] = None,
        cycle_date: date = None
    ) -> Dict[str, Any]:
        """
        Run complete monitoring cycle: evaluate performance, detect drift, and retrain if needed.
        
        Args:
            holdout_features: Holdout dataset for performance evaluation
            holdout_labels: Holdout dataset labels
            current_features: Current production features for drift detection
            training_features: New training data (required if retraining is triggered)
            training_labels: New training labels (required if retraining is triggered)
            cycle_date: Date of monitoring cycle (defaults to today)
            
        Returns:
            Dictionary with complete monitoring cycle results
        """
        if cycle_date is None:
            cycle_date = date.today()
        
        logger.info(f"Running monitoring cycle for {cycle_date}")
        
        cycle_results = {
            'cycle_date': cycle_date.isoformat(),
            'performance_evaluation': None,
            'drift_detection': None,
            'retraining_triggered': False,
            'retraining_results': None
        }
        
        try:
            # 1. Evaluate model performance
            performance_results = self.performance_monitor.evaluate_daily_accuracy(
                holdout_features, holdout_labels, cycle_date
            )
            cycle_results['performance_evaluation'] = performance_results
            
            # 2. Detect feature drift
            if current_features:
                # Set baseline if not already set
                if not self.drift_detector.baseline_distributions:
                    self.drift_detector.set_baseline_distribution(current_features, cycle_date)
                    drift_results = None  # Skip drift detection on first run
                else:
                    drift_results = self.drift_detector.detect_drift(current_features, cycle_date)
                    cycle_results['drift_detection'] = drift_results
            else:
                drift_results = None
            
            # 3. Check if retraining is needed
            should_retrain, reasons = self.should_retrain_model(performance_results, drift_results)
            
            # 4. Trigger retraining if needed
            if should_retrain:
                if training_features and training_labels:
                    retraining_results = self.trigger_retraining(
                        training_features, training_labels, reasons, cycle_date
                    )
                    cycle_results['retraining_triggered'] = True
                    cycle_results['retraining_results'] = retraining_results
                else:
                    logger.warning("Retraining needed but no training data provided")
                    cycle_results['retraining_triggered'] = False
                    cycle_results['retraining_results'] = {
                        'error': 'No training data provided for retraining'
                    }
            
            logger.info(f"Monitoring cycle completed. Retraining triggered: {should_retrain}")
            return cycle_results
            
        except Exception as e:
            logger.error(f"Error in monitoring cycle: {str(e)}")
            cycle_results['error'] = str(e)
            return cycle_results