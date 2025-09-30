"""
Model training performance tests with realistic data volumes.
Tests ML pipeline performance, memory usage, and training time benchmarks.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
import time
import psutil
import gc
import tempfile
import joblib
from typing import List, Dict, Any
from sklearn.metrics import accuracy_score, roc_auc_score

from src.services.ml_pipeline import ChurnPredictor, FeatureEngineer, ModelEvaluator
from src.models.churn_features import ChurnFeatures


class TestModelTrainingPerformance:
    """Performance tests for ML model training with realistic data volumes."""
    
    @pytest.fixture
    def performance_datasets(self):
        """Define dataset sizes for performance testing."""
        return {
            'small': 1000,      # 1K players
            'medium': 10000,    # 10K players  
            'large': 50000,     # 50K players (reduced from 100K for CI performance)
            'xlarge': 100000    # 100K players (for stress testing)
        }
    
    def generate_realistic_churn_features(self, num_players: int, seed: int = 42) -> List[ChurnFeatures]:
        """Generate realistic churn features with proper distributions."""
        np.random.seed(seed)
        features = []
        base_date = date(2024, 1, 1)
        
        for i in range(num_players):
            # Create realistic feature distributions
            
            # Days since last session (exponential distribution)
            days_since_last = int(np.random.exponential(5))  # Most players active recently
            days_since_last = min(days_since_last, 30)  # Cap at 30 days
            
            # Sessions last 7 days (Poisson distribution) - must respect business logic
            if days_since_last > 7:
                sessions_7d = 0  # No sessions if last session was more than 7 days ago
            else:
                sessions_7d = int(np.random.poisson(3))  # Average 3 sessions per week
                sessions_7d = max(0, min(sessions_7d, 21))  # Cap at 3 per day
            
            # Average session duration (log-normal distribution)
            avg_duration = np.random.lognormal(3.5, 0.8)  # ~30 min average
            avg_duration = max(5, min(avg_duration, 180))  # 5 min to 3 hours
            
            # Levels completed (related to engagement)
            engagement_factor = np.random.beta(2, 5)  # Most players low-medium engagement
            levels_completed = int(engagement_factor * 50)
            
            # Purchases (zero-inflated log-normal)
            if np.random.random() < 0.7:  # 70% non-paying users
                purchases = 0.0
            else:
                purchases = np.random.lognormal(2.0, 1.0)  # ~$7 average for paying users
                purchases = min(purchases, 500)  # Cap at $500
            
            # Social connections (Poisson)
            social_connections = int(np.random.poisson(5))  # Average 5 connections
            social_connections = min(social_connections, 100)
            
            feature = ChurnFeatures(
                player_id=f"player_{i:06d}",
                days_since_last_session=days_since_last,
                sessions_last_7_days=sessions_7d,
                avg_session_duration_minutes=avg_duration,
                levels_completed_last_week=levels_completed,
                purchases_last_30_days=purchases,
                social_connections=social_connections,
                feature_date=base_date + timedelta(days=i % 30)
            )
            features.append(feature)
        
        return features
    
    def generate_realistic_churn_labels(self, churn_features: List[ChurnFeatures]) -> List[bool]:
        """Generate realistic churn labels based on feature patterns."""
        labels = []
        
        for feature in churn_features:
            # Calculate churn probability based on realistic factors
            churn_prob = 0.1  # Base churn rate
            
            # Recency factor (more days = higher churn probability)
            churn_prob += feature.days_since_last_session * 0.02
            
            # Frequency factor (fewer sessions = higher churn)
            churn_prob += max(0, (7 - feature.sessions_last_7_days)) * 0.03
            
            # Engagement factor (shorter sessions = higher churn)
            if feature.avg_session_duration_minutes < 15:
                churn_prob += 0.15
            elif feature.avg_session_duration_minutes < 30:
                churn_prob += 0.05
            
            # Monetary factor (no purchases = higher churn)
            if feature.purchases_last_30_days == 0:
                churn_prob += 0.1
            else:
                churn_prob -= 0.05  # Paying users less likely to churn
            
            # Social factor (fewer connections = higher churn)
            if feature.social_connections == 0:
                churn_prob += 0.1
            elif feature.social_connections > 10:
                churn_prob -= 0.05
            
            # Cap probability
            churn_prob = max(0.01, min(churn_prob, 0.8))
            
            # Generate label
            is_churned = np.random.random() < churn_prob
            labels.append(is_churned)
        
        return labels
    
    @pytest.mark.performance
    def test_feature_engineering_scalability(self, performance_datasets):
        """Test feature engineering performance across different dataset sizes."""
        feature_engineer = FeatureEngineer()
        results = {}
        
        for dataset_name, size in performance_datasets.items():
            if dataset_name == 'xlarge':  # Skip extra large for regular tests
                continue
                
            print(f"\nTesting feature engineering with {dataset_name} dataset ({size:,} players)")
            
            # Generate data
            churn_features = self.generate_realistic_churn_features(size)
            
            # Measure performance
            start_time = time.time()
            memory_before = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            features_df = feature_engineer.engineer_features(churn_features)
            X = feature_engineer.get_feature_matrix(features_df)
            
            end_time = time.time()
            memory_after = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            duration = end_time - start_time
            memory_used = memory_after - memory_before
            throughput = size / duration  # players per second
            
            results[dataset_name] = {
                'duration': duration,
                'memory_mb': memory_used,
                'throughput': throughput,
                'features_shape': X.shape
            }
            
            # Performance assertions based on dataset size
            if dataset_name == 'small':
                assert duration < 5.0, f"Small dataset took {duration:.2f}s, expected < 5.0s"
                assert memory_used < 50, f"Small dataset used {memory_used:.1f}MB, expected < 50MB"
            elif dataset_name == 'medium':
                assert duration < 30.0, f"Medium dataset took {duration:.2f}s, expected < 30.0s"
                assert memory_used < 200, f"Medium dataset used {memory_used:.1f}MB, expected < 200MB"
            elif dataset_name == 'large':
                assert duration < 120.0, f"Large dataset took {duration:.2f}s, expected < 120.0s"
                assert memory_used < 500, f"Large dataset used {memory_used:.1f}MB, expected < 500MB"
            
            print(f"  Duration: {duration:.2f}s")
            print(f"  Memory: {memory_used:.1f}MB")
            print(f"  Throughput: {throughput:.0f} players/second")
            print(f"  Features shape: {X.shape}")
            
            # Clean up memory
            del churn_features, features_df, X
            gc.collect()
        
        # Test scalability (should be roughly linear)
        if 'small' in results and 'medium' in results:
            small_throughput = results['small']['throughput']
            medium_throughput = results['medium']['throughput']
            
            # Throughput should not degrade significantly with larger datasets
            throughput_ratio = medium_throughput / small_throughput
            assert throughput_ratio > 0.5, f"Throughput degraded significantly: {throughput_ratio:.2f}"
    
    @pytest.mark.performance
    def test_model_training_scalability(self, performance_datasets):
        """Test model training performance across different dataset sizes."""
        results = {}
        
        for dataset_name, size in performance_datasets.items():
            if dataset_name in ['large', 'xlarge']:  # Skip large datasets for training tests
                continue
                
            print(f"\nTesting model training with {dataset_name} dataset ({size:,} players)")
            
            # Generate data
            churn_features = self.generate_realistic_churn_features(size)
            labels = self.generate_realistic_churn_labels(churn_features)
            
            # Initialize predictor with reduced hyperparameter grid for performance
            predictor = ChurnPredictor()
            predictor.param_grids = {
                'random_forest': {
                    'n_estimators': [50, 100],
                    'max_depth': [10, None],
                    'min_samples_split': [2]
                },
                'gradient_boosting': {
                    'n_estimators': [50, 100],
                    'max_depth': [3, 5],
                    'learning_rate': [0.1]
                }
            }
            
            # Measure performance
            start_time = time.time()
            memory_before = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            X, y = predictor.prepare_training_data(churn_features, labels)
            training_results = predictor.train_models(X, y, cv_folds=3)
            
            end_time = time.time()
            memory_after = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            duration = end_time - start_time
            memory_used = memory_after - memory_before
            
            # Get best model performance
            best_model_metrics = training_results[predictor.best_model_name]['metrics']
            
            results[dataset_name] = {
                'duration': duration,
                'memory_mb': memory_used,
                'accuracy': best_model_metrics['accuracy'],
                'roc_auc': best_model_metrics['roc_auc'],
                'best_model': predictor.best_model_name
            }
            
            # Performance assertions
            if dataset_name == 'small':
                assert duration < 60.0, f"Small dataset training took {duration:.2f}s, expected < 60.0s"
                assert memory_used < 200, f"Small dataset used {memory_used:.1f}MB, expected < 200MB"
            elif dataset_name == 'medium':
                assert duration < 300.0, f"Medium dataset training took {duration:.2f}s, expected < 300.0s"
                assert memory_used < 500, f"Medium dataset used {memory_used:.1f}MB, expected < 500MB"
            
            # Model quality assertions (requirement: >= 80% accuracy)
            assert best_model_metrics['accuracy'] >= 0.8, f"Model accuracy {best_model_metrics['accuracy']:.3f} < 0.8"
            assert best_model_metrics['roc_auc'] >= 0.7, f"Model ROC-AUC {best_model_metrics['roc_auc']:.3f} < 0.7"
            
            print(f"  Duration: {duration:.2f}s")
            print(f"  Memory: {memory_used:.1f}MB")
            print(f"  Best model: {predictor.best_model_name}")
            print(f"  Accuracy: {best_model_metrics['accuracy']:.3f}")
            print(f"  ROC-AUC: {best_model_metrics['roc_auc']:.3f}")
            
            # Clean up memory
            del churn_features, labels, X, y, predictor
            gc.collect()
    
    @pytest.mark.performance
    def test_model_prediction_scalability(self):
        """Test model prediction performance with various batch sizes."""
        # Train a model first
        print("\nTraining model for prediction performance testing...")
        training_size = 2000
        churn_features = self.generate_realistic_churn_features(training_size)
        labels = self.generate_realistic_churn_labels(churn_features)
        
        predictor = ChurnPredictor()
        # Use minimal hyperparameter grid for fast training
        predictor.param_grids = {
            'random_forest': {
                'n_estimators': [50],
                'max_depth': [10]
            }
        }
        
        X_train, y_train = predictor.prepare_training_data(churn_features, labels)
        predictor.train_models(X_train, y_train, cv_folds=3)
        
        # Test prediction performance with different batch sizes
        batch_sizes = [100, 500, 1000, 5000, 10000]
        results = {}
        
        for batch_size in batch_sizes:
            print(f"\nTesting prediction with batch size: {batch_size:,}")
            
            # Generate prediction data
            prediction_features = self.generate_realistic_churn_features(batch_size, seed=123)
            
            # Measure prediction performance
            start_time = time.time()
            memory_before = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            probabilities = predictor.predict_churn_probability(prediction_features)
            binary_predictions = predictor.predict_churn_binary(prediction_features)
            
            end_time = time.time()
            memory_after = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            duration = end_time - start_time
            memory_used = memory_after - memory_before
            throughput = batch_size / duration  # predictions per second
            
            results[batch_size] = {
                'duration': duration,
                'memory_mb': memory_used,
                'throughput': throughput
            }
            
            # Performance assertions
            assert throughput > 500, f"Prediction throughput {throughput:.0f}/s < 500/s for batch size {batch_size}"
            assert len(probabilities) == batch_size
            assert len(binary_predictions) == batch_size
            assert all(0 <= p <= 1 for p in probabilities), "Probabilities should be between 0 and 1"
            
            print(f"  Duration: {duration:.3f}s")
            print(f"  Memory: {memory_used:.1f}MB")
            print(f"  Throughput: {throughput:.0f} predictions/second")
            
            # Clean up
            del prediction_features, probabilities, binary_predictions
            gc.collect()
        
        # Test that throughput scales reasonably
        small_throughput = results[100]['throughput']
        large_throughput = results[10000]['throughput']
        
        # Large batches should be more efficient (higher throughput)
        assert large_throughput >= small_throughput * 0.8, "Prediction throughput should scale with batch size"
    
    @pytest.mark.performance
    def test_model_memory_efficiency(self):
        """Test model memory usage and efficiency."""
        print("\nTesting model memory efficiency...")
        
        # Test with progressively larger datasets
        sizes = [1000, 2000, 5000]
        memory_usage = []
        
        for size in sizes:
            print(f"\nTesting memory usage with {size:,} players")
            
            # Force garbage collection before test
            gc.collect()
            memory_baseline = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            # Generate data
            churn_features = self.generate_realistic_churn_features(size)
            labels = self.generate_realistic_churn_labels(churn_features)
            
            # Train model
            predictor = ChurnPredictor()
            predictor.param_grids = {
                'random_forest': {'n_estimators': [50], 'max_depth': [10]}
            }
            
            X, y = predictor.prepare_training_data(churn_features, labels)
            predictor.train_models(X, y, cv_folds=3)
            
            memory_peak = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            memory_used = memory_peak - memory_baseline
            memory_per_sample = memory_used / size
            
            memory_usage.append({
                'size': size,
                'memory_mb': memory_used,
                'memory_per_sample': memory_per_sample
            })
            
            print(f"  Memory used: {memory_used:.1f}MB")
            print(f"  Memory per sample: {memory_per_sample:.3f}MB")
            
            # Memory efficiency assertions
            assert memory_per_sample < 0.1, f"Memory per sample {memory_per_sample:.3f}MB too high"
            
            # Clean up
            del churn_features, labels, X, y, predictor
            gc.collect()
        
        # Test memory scaling (should be roughly linear)
        if len(memory_usage) >= 2:
            ratio_size = memory_usage[1]['size'] / memory_usage[0]['size']
            ratio_memory = memory_usage[1]['memory_mb'] / memory_usage[0]['memory_mb']
            
            # Memory usage should scale roughly linearly with data size
            assert ratio_memory < ratio_size * 2, f"Memory scaling too steep: {ratio_memory:.2f} vs {ratio_size:.2f}"
    
    @pytest.mark.performance
    def test_model_serialization_performance(self):
        """Test model saving and loading performance."""
        print("\nTesting model serialization performance...")
        
        # Train a model
        size = 2000
        churn_features = self.generate_realistic_churn_features(size)
        labels = self.generate_realistic_churn_labels(churn_features)
        
        predictor = ChurnPredictor()
        predictor.param_grids = {
            'random_forest': {'n_estimators': [100], 'max_depth': [10]}
        }
        
        X, y = predictor.prepare_training_data(churn_features, labels)
        predictor.train_models(X, y, cv_folds=3)
        
        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as tmp_file:
            model_path = tmp_file.name
        
        try:
            # Test model saving performance
            start_time = time.time()
            predictor.save_model(model_path)
            save_duration = time.time() - start_time
            
            # Test model loading performance
            new_predictor = ChurnPredictor()
            start_time = time.time()
            new_predictor.load_model(model_path)
            load_duration = time.time() - start_time
            
            # Performance assertions
            assert save_duration < 5.0, f"Model saving took {save_duration:.2f}s, expected < 5.0s"
            assert load_duration < 2.0, f"Model loading took {load_duration:.2f}s, expected < 2.0s"
            
            # Verify loaded model works
            test_features = self.generate_realistic_churn_features(10)
            predictions = new_predictor.predict_churn_probability(test_features)
            assert len(predictions) == 10
            
            print(f"  Model save time: {save_duration:.3f}s")
            print(f"  Model load time: {load_duration:.3f}s")
            
        finally:
            # Clean up
            import os
            if os.path.exists(model_path):
                os.unlink(model_path)
    
    @pytest.mark.performance
    @pytest.mark.slow
    def test_stress_test_large_dataset(self, performance_datasets):
        """Stress test with very large dataset (only run when explicitly requested)."""
        size = performance_datasets['xlarge']  # 100K players
        print(f"\nStress testing with {size:,} players...")
        
        # Generate data in chunks to avoid memory issues
        chunk_size = 10000
        all_features = []
        
        for i in range(0, size, chunk_size):
            chunk_features = self.generate_realistic_churn_features(
                min(chunk_size, size - i), 
                seed=i
            )
            all_features.extend(chunk_features)
            
            if i % 50000 == 0:
                print(f"  Generated {i + len(chunk_features):,} features...")
        
        # Test feature engineering only (skip training for stress test)
        feature_engineer = FeatureEngineer()
        
        start_time = time.time()
        memory_before = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # Process in chunks to manage memory
        chunk_size = 20000
        all_X = []
        
        for i in range(0, len(all_features), chunk_size):
            chunk = all_features[i:i + chunk_size]
            features_df = feature_engineer.engineer_features(chunk)
            X_chunk = feature_engineer.get_feature_matrix(features_df)
            all_X.append(X_chunk)
            
            if i % 50000 == 0:
                print(f"  Processed {i + len(chunk):,} features...")
            
            # Clean up chunk data
            del chunk, features_df, X_chunk
            gc.collect()
        
        # Combine results
        X_final = np.vstack(all_X)
        
        end_time = time.time()
        memory_after = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        duration = end_time - start_time
        memory_used = memory_after - memory_before
        throughput = size / duration
        
        # Stress test assertions
        assert duration < 600.0, f"Stress test took {duration:.2f}s, expected < 600.0s (10 min)"
        assert memory_used < 2000, f"Stress test used {memory_used:.1f}MB, expected < 2000MB"
        assert X_final.shape[0] == size
        
        print(f"  Duration: {duration:.2f}s")
        print(f"  Memory: {memory_used:.1f}MB")
        print(f"  Throughput: {throughput:.0f} players/second")
        print(f"  Final shape: {X_final.shape}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "-m", "performance"])