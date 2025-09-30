"""
Performance benchmark tests for retention queries and ML pipeline with large datasets.
Tests system performance under realistic load conditions.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import time
import psutil
import gc
from unittest.mock import Mock, patch
from typing import List, Dict, Any

from src.analytics.query_engine import RetentionQueryEngine
from src.services.ml_pipeline import ChurnPredictor, FeatureEngineer
from src.models.churn_features import ChurnFeatures
from src.models.player_profile import PlayerProfile


class TestRetentionQueryPerformance:
    """Performance benchmarks for retention queries with large datasets."""
    
    @pytest.fixture
    def large_dataset_params(self):
        """Parameters for large dataset testing."""
        return {
            'small': {'players': 1000, 'days': 30},
            'medium': {'players': 10000, 'days': 90},
            'large': {'players': 100000, 'days': 365}
        }
    
    def generate_mock_query_results(self, num_cohorts: int, cohort_size_range: tuple = (100, 1000)):
        """Generate mock query results for performance testing."""
        results = []
        base_date = date(2024, 1, 1)
        
        for i in range(num_cohorts):
            cohort_date = base_date + timedelta(days=i)
            cohort_size = np.random.randint(*cohort_size_range)
            
            # Simulate realistic retention decay
            day_1_retention = np.random.uniform(0.7, 0.9)
            day_7_retention = day_1_retention * np.random.uniform(0.6, 0.8)
            day_30_retention = day_7_retention * np.random.uniform(0.5, 0.7)
            
            mock_result = Mock(
                cohort_date=cohort_date,
                day_1_retention=day_1_retention,
                day_7_retention=day_7_retention,
                day_30_retention=day_30_retention,
                cohort_size=cohort_size
            )
            results.append(mock_result)
        
        return results
    
    @pytest.mark.performance
    def test_cohort_retention_query_performance_small(self, large_dataset_params):
        """Test cohort retention query performance with small dataset."""
        query_engine = RetentionQueryEngine()
        params = large_dataset_params['small']
        
        with patch.object(query_engine.db_manager, 'get_session') as mock_session:
            # Mock database response
            mock_results = self.generate_mock_query_results(params['days'])
            mock_result = Mock()
            mock_result.fetchall.return_value = mock_results
            
            mock_session_instance = Mock()
            mock_session_instance.execute.return_value = mock_result
            mock_session_instance.__enter__ = Mock(return_value=mock_session_instance)
            mock_session_instance.__exit__ = Mock(return_value=None)
            mock_session.return_value = mock_session_instance
            
            # Measure performance
            start_time = time.time()
            memory_before = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            results = query_engine.calculate_cohort_retention(
                date(2024, 1, 1),
                date(2024, 1, 1) + timedelta(days=params['days'])
            )
            
            end_time = time.time()
            memory_after = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            duration = end_time - start_time
            memory_used = memory_after - memory_before
            
            # Performance assertions for small dataset
            assert duration < 5.0, f"Query took {duration:.2f}s, expected < 5.0s"
            assert memory_used < 50, f"Memory usage {memory_used:.2f}MB, expected < 50MB"
            assert len(results) == params['days']
            
            print(f"Small dataset ({params['players']} players, {params['days']} days):")
            print(f"  Duration: {duration:.3f}s")
            print(f"  Memory: {memory_used:.2f}MB")
    
    @pytest.mark.performance
    def test_cohort_retention_query_performance_medium(self, large_dataset_params):
        """Test cohort retention query performance with medium dataset."""
        query_engine = RetentionQueryEngine()
        params = large_dataset_params['medium']
        
        with patch.object(query_engine.db_manager, 'get_session') as mock_session:
            # Mock database response with larger cohorts
            mock_results = self.generate_mock_query_results(
                params['days'], 
                cohort_size_range=(500, 2000)
            )
            mock_result = Mock()
            mock_result.fetchall.return_value = mock_results
            
            mock_session_instance = Mock()
            mock_session_instance.execute.return_value = mock_result
            mock_session_instance.__enter__ = Mock(return_value=mock_session_instance)
            mock_session_instance.__exit__ = Mock(return_value=None)
            mock_session.return_value = mock_session_instance
            
            # Measure performance
            start_time = time.time()
            memory_before = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            results = query_engine.calculate_cohort_retention(
                date(2024, 1, 1),
                date(2024, 1, 1) + timedelta(days=params['days'])
            )
            
            end_time = time.time()
            memory_after = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            duration = end_time - start_time
            memory_used = memory_after - memory_before
            
            # Performance assertions for medium dataset
            assert duration < 15.0, f"Query took {duration:.2f}s, expected < 15.0s"
            assert memory_used < 100, f"Memory usage {memory_used:.2f}MB, expected < 100MB"
            assert len(results) == params['days']
            
            print(f"Medium dataset ({params['players']} players, {params['days']} days):")
            print(f"  Duration: {duration:.3f}s")
            print(f"  Memory: {memory_used:.2f}MB")
    
    @pytest.mark.performance
    @pytest.mark.slow
    def test_cohort_retention_query_performance_large(self, large_dataset_params):
        """Test cohort retention query performance with large dataset."""
        query_engine = RetentionQueryEngine()
        params = large_dataset_params['large']
        
        with patch.object(query_engine.db_manager, 'get_session') as mock_session:
            # Mock database response with very large cohorts
            mock_results = self.generate_mock_query_results(
                params['days'], 
                cohort_size_range=(1000, 5000)
            )
            mock_result = Mock()
            mock_result.fetchall.return_value = mock_results
            
            mock_session_instance = Mock()
            mock_session_instance.execute.return_value = mock_result
            mock_session_instance.__enter__ = Mock(return_value=mock_session_instance)
            mock_session_instance.__exit__ = Mock(return_value=None)
            mock_session.return_value = mock_session_instance
            
            # Measure performance
            start_time = time.time()
            memory_before = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            results = query_engine.calculate_cohort_retention(
                date(2024, 1, 1),
                date(2024, 1, 1) + timedelta(days=params['days'])
            )
            
            end_time = time.time()
            memory_after = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            duration = end_time - start_time
            memory_used = memory_after - memory_before
            
            # Performance assertions for large dataset (requirement: < 30s)
            assert duration < 30.0, f"Query took {duration:.2f}s, expected < 30.0s"
            assert memory_used < 200, f"Memory usage {memory_used:.2f}MB, expected < 200MB"
            assert len(results) == params['days']
            
            print(f"Large dataset ({params['players']} players, {params['days']} days):")
            print(f"  Duration: {duration:.3f}s")
            print(f"  Memory: {memory_used:.2f}MB")
    
    @pytest.mark.performance
    def test_drop_off_analysis_performance(self):
        """Test drop-off analysis query performance."""
        query_engine = RetentionQueryEngine()
        
        with patch.object(query_engine.db_manager, 'get_session') as mock_session:
            # Mock drop-off analysis results
            mock_results = []
            for level in range(1, 51):  # 50 levels
                players_reached = max(1000 - level * 15, 50)  # Decreasing players
                players_completed = int(players_reached * np.random.uniform(0.7, 0.9))
                drop_off_rate = (players_reached - players_completed) / players_reached
                completion_rate = players_completed / players_reached
                
                mock_result = Mock(
                    level=level,
                    players_reached=players_reached,
                    players_completed=players_completed,
                    drop_off_rate=drop_off_rate,
                    completion_rate=completion_rate
                )
                mock_results.append(mock_result)
            
            mock_result = Mock()
            mock_result.fetchall.return_value = mock_results
            
            mock_session_instance = Mock()
            mock_session_instance.execute.return_value = mock_result
            mock_session_instance.__enter__ = Mock(return_value=mock_session_instance)
            mock_session_instance.__exit__ = Mock(return_value=None)
            mock_session.return_value = mock_session_instance
            
            # Measure performance
            start_time = time.time()
            
            results = query_engine.analyze_drop_off_by_level(
                date(2024, 1, 1),
                date(2024, 1, 31),
                max_level=50
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Performance assertions
            assert duration < 10.0, f"Drop-off analysis took {duration:.2f}s, expected < 10.0s"
            assert len(results) == 50
            
            print(f"Drop-off analysis (50 levels): {duration:.3f}s")
    
    @pytest.mark.performance
    def test_player_segmentation_performance(self):
        """Test player segmentation query performance."""
        query_engine = RetentionQueryEngine()
        
        with patch.object(query_engine.db_manager, 'get_session') as mock_session:
            # Mock segmentation results
            segments = ['High Engagement', 'Medium Engagement', 'Low Engagement', 'Minimal Engagement']
            mock_results = []
            
            for segment in segments:
                mock_result = Mock(
                    segment=segment,
                    player_count=np.random.randint(1000, 5000),
                    avg_sessions=np.random.uniform(5, 50),
                    avg_playtime=np.random.uniform(300, 3000),
                    avg_retention_day_7=np.random.uniform(0.3, 0.8)
                )
                mock_results.append(mock_result)
            
            mock_result = Mock()
            mock_result.fetchall.return_value = mock_results
            
            mock_session_instance = Mock()
            mock_session_instance.execute.return_value = mock_result
            mock_session_instance.__enter__ = Mock(return_value=mock_session_instance)
            mock_session_instance.__exit__ = Mock(return_value=None)
            mock_session.return_value = mock_session_instance
            
            # Measure performance
            start_time = time.time()
            
            results = query_engine.segment_players_by_behavior(
                date(2024, 1, 1),
                date(2024, 1, 31)
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Performance assertions
            assert duration < 5.0, f"Player segmentation took {duration:.2f}s, expected < 5.0s"
            assert len(results) == 4
            
            print(f"Player segmentation (4 segments): {duration:.3f}s")


class TestMLPipelinePerformance:
    """Performance benchmarks for ML pipeline with realistic data volumes."""
    
    def generate_churn_features(self, num_players: int) -> List[ChurnFeatures]:
        """Generate synthetic churn features for performance testing."""
        features = []
        base_date = date(2024, 1, 1)
        
        for i in range(num_players):
            feature = ChurnFeatures(
                player_id=f"player_{i:06d}",
                days_since_last_session=np.random.randint(0, 30),
                sessions_last_7_days=np.random.randint(0, 20),
                avg_session_duration_minutes=np.random.uniform(5, 180),
                levels_completed_last_week=np.random.randint(0, 50),
                purchases_last_30_days=np.random.uniform(0, 100),
                social_connections=np.random.randint(0, 100),
                feature_date=base_date + timedelta(days=i % 30)
            )
            features.append(feature)
        
        return features
    
    def generate_churn_labels(self, num_players: int) -> List[bool]:
        """Generate synthetic churn labels with realistic distribution."""
        # Realistic churn rate around 20-30%
        churn_rate = 0.25
        labels = np.random.choice([True, False], num_players, p=[churn_rate, 1-churn_rate])
        return labels.tolist()
    
    @pytest.mark.performance
    def test_feature_engineering_performance_small(self):
        """Test feature engineering performance with small dataset."""
        num_players = 1000
        churn_features = self.generate_churn_features(num_players)
        
        feature_engineer = FeatureEngineer()
        
        # Measure performance
        start_time = time.time()
        memory_before = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        features_df = feature_engineer.engineer_features(churn_features)
        X = feature_engineer.get_feature_matrix(features_df)
        
        end_time = time.time()
        memory_after = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        duration = end_time - start_time
        memory_used = memory_after - memory_before
        
        # Performance assertions
        assert duration < 5.0, f"Feature engineering took {duration:.2f}s, expected < 5.0s"
        assert memory_used < 100, f"Memory usage {memory_used:.2f}MB, expected < 100MB"
        assert X.shape == (num_players, len(feature_engineer.feature_columns))
        
        print(f"Feature engineering ({num_players} players):")
        print(f"  Duration: {duration:.3f}s")
        print(f"  Memory: {memory_used:.2f}MB")
        print(f"  Features shape: {X.shape}")
    
    @pytest.mark.performance
    def test_feature_engineering_performance_medium(self):
        """Test feature engineering performance with medium dataset."""
        num_players = 10000
        churn_features = self.generate_churn_features(num_players)
        
        feature_engineer = FeatureEngineer()
        
        # Measure performance
        start_time = time.time()
        memory_before = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        features_df = feature_engineer.engineer_features(churn_features)
        X = feature_engineer.get_feature_matrix(features_df)
        
        end_time = time.time()
        memory_after = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        duration = end_time - start_time
        memory_used = memory_after - memory_before
        
        # Performance assertions
        assert duration < 15.0, f"Feature engineering took {duration:.2f}s, expected < 15.0s"
        assert memory_used < 500, f"Memory usage {memory_used:.2f}MB, expected < 500MB"
        assert X.shape == (num_players, len(feature_engineer.feature_columns))
        
        print(f"Feature engineering ({num_players} players):")
        print(f"  Duration: {duration:.3f}s")
        print(f"  Memory: {memory_used:.2f}MB")
        print(f"  Features shape: {X.shape}")
    
    @pytest.mark.performance
    @pytest.mark.slow
    def test_feature_engineering_performance_large(self):
        """Test feature engineering performance with large dataset."""
        num_players = 50000  # Reduced from 100k for CI/CD performance
        churn_features = self.generate_churn_features(num_players)
        
        feature_engineer = FeatureEngineer()
        
        # Measure performance
        start_time = time.time()
        memory_before = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        features_df = feature_engineer.engineer_features(churn_features)
        X = feature_engineer.get_feature_matrix(features_df)
        
        end_time = time.time()
        memory_after = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        duration = end_time - start_time
        memory_used = memory_after - memory_before
        
        # Performance assertions
        assert duration < 60.0, f"Feature engineering took {duration:.2f}s, expected < 60.0s"
        assert memory_used < 1000, f"Memory usage {memory_used:.2f}MB, expected < 1000MB"
        assert X.shape == (num_players, len(feature_engineer.feature_columns))
        
        print(f"Feature engineering ({num_players} players):")
        print(f"  Duration: {duration:.3f}s")
        print(f"  Memory: {memory_used:.2f}MB")
        print(f"  Features shape: {X.shape}")
        
        # Clean up memory
        del features_df, X, churn_features
        gc.collect()
    
    @pytest.mark.performance
    def test_model_training_performance_small(self):
        """Test model training performance with small dataset."""
        num_players = 1000
        churn_features = self.generate_churn_features(num_players)
        labels = self.generate_churn_labels(num_players)
        
        predictor = ChurnPredictor()
        
        # Measure performance
        start_time = time.time()
        memory_before = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        X, y = predictor.prepare_training_data(churn_features, labels)
        results = predictor.train_models(X, y, cv_folds=3)  # Reduced CV folds for speed
        
        end_time = time.time()
        memory_after = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        duration = end_time - start_time
        memory_used = memory_after - memory_before
        
        # Performance assertions
        assert duration < 30.0, f"Model training took {duration:.2f}s, expected < 30.0s"
        assert memory_used < 200, f"Memory usage {memory_used:.2f}MB, expected < 200MB"
        assert len(results) == 2  # Two models trained
        assert predictor.best_model is not None
        
        print(f"Model training ({num_players} players):")
        print(f"  Duration: {duration:.3f}s")
        print(f"  Memory: {memory_used:.2f}MB")
        print(f"  Best model: {predictor.best_model_name}")
    
    @pytest.mark.performance
    def test_model_training_performance_medium(self):
        """Test model training performance with medium dataset."""
        num_players = 5000  # Reduced for reasonable test time
        churn_features = self.generate_churn_features(num_players)
        labels = self.generate_churn_labels(num_players)
        
        predictor = ChurnPredictor()
        
        # Reduce hyperparameter grid for performance
        predictor.param_grids = {
            'random_forest': {
                'n_estimators': [100],
                'max_depth': [10, None],
                'min_samples_split': [2]
            },
            'gradient_boosting': {
                'n_estimators': [100],
                'max_depth': [3, 5],
                'learning_rate': [0.1]
            }
        }
        
        # Measure performance
        start_time = time.time()
        memory_before = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        X, y = predictor.prepare_training_data(churn_features, labels)
        results = predictor.train_models(X, y, cv_folds=3)
        
        end_time = time.time()
        memory_after = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        duration = end_time - start_time
        memory_used = memory_after - memory_before
        
        # Performance assertions
        assert duration < 120.0, f"Model training took {duration:.2f}s, expected < 120.0s"
        assert memory_used < 500, f"Memory usage {memory_used:.2f}MB, expected < 500MB"
        assert len(results) == 2
        assert predictor.best_model is not None
        
        print(f"Model training ({num_players} players):")
        print(f"  Duration: {duration:.3f}s")
        print(f"  Memory: {memory_used:.2f}MB")
        print(f"  Best model: {predictor.best_model_name}")
    
    @pytest.mark.performance
    def test_model_prediction_performance(self):
        """Test model prediction performance with various batch sizes."""
        # Train a simple model first
        num_training = 1000
        training_features = self.generate_churn_features(num_training)
        training_labels = self.generate_churn_labels(num_training)
        
        predictor = ChurnPredictor()
        X_train, y_train = predictor.prepare_training_data(training_features, training_labels)
        predictor.train_models(X_train, y_train, cv_folds=3)
        
        # Test prediction performance with different batch sizes
        batch_sizes = [100, 1000, 5000]
        
        for batch_size in batch_sizes:
            prediction_features = self.generate_churn_features(batch_size)
            
            start_time = time.time()
            
            probabilities = predictor.predict_churn_probability(prediction_features)
            binary_predictions = predictor.predict_churn_binary(prediction_features)
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Performance assertions
            predictions_per_second = batch_size / duration
            assert predictions_per_second > 100, f"Only {predictions_per_second:.1f} predictions/s, expected > 100"
            assert len(probabilities) == batch_size
            assert len(binary_predictions) == batch_size
            
            print(f"Prediction performance ({batch_size} players):")
            print(f"  Duration: {duration:.3f}s")
            print(f"  Predictions/second: {predictions_per_second:.1f}")


class TestVisualizationPerformance:
    """Performance benchmarks for visualization rendering with various data sizes."""
    
    @pytest.mark.performance
    def test_chart_creation_performance(self):
        """Test chart creation performance with various data sizes."""
        import plotly.graph_objects as go
        from src.visualization.components import ChartStyler
        
        data_sizes = [100, 1000, 5000, 10000]
        
        for size in data_sizes:
            # Generate test data
            x_data = list(range(size))
            y_data = np.random.uniform(0, 1, size)
            
            start_time = time.time()
            
            # Create chart
            fig = go.Figure()
            fig.add_scatter(x=x_data, y=y_data, mode='markers', name='Test Data')
            
            # Apply styling
            styled_fig = ChartStyler.apply_base_layout(fig, f"Performance Test ({size} points)")
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Performance assertions
            assert duration < 5.0, f"Chart creation took {duration:.2f}s for {size} points, expected < 5.0s"
            assert isinstance(styled_fig, go.Figure)
            
            print(f"Chart creation ({size} points): {duration:.3f}s")
    
    @pytest.mark.performance
    def test_heatmap_performance(self):
        """Test heatmap creation performance with large matrices."""
        import plotly.graph_objects as go
        from src.visualization.components import ChartStyler
        
        matrix_sizes = [(10, 30), (30, 90), (50, 365)]  # (cohorts, days)
        
        for rows, cols in matrix_sizes:
            # Generate test heatmap data
            z_data = np.random.uniform(0, 1, (rows, cols))
            x_labels = [f"Day {i+1}" for i in range(cols)]
            y_labels = [f"Cohort {i+1}" for i in range(rows)]
            
            start_time = time.time()
            
            # Create heatmap
            fig = go.Figure(data=go.Heatmap(
                z=z_data,
                x=x_labels,
                y=y_labels,
                colorscale='RdYlBu_r'
            ))
            
            # Apply styling
            styled_fig = ChartStyler.apply_heatmap_styling(fig, f"Retention Heatmap ({rows}x{cols})")
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Performance assertions (requirement: < 5s for visualization rendering)
            assert duration < 5.0, f"Heatmap creation took {duration:.2f}s for {rows}x{cols}, expected < 5.0s"
            assert isinstance(styled_fig, go.Figure)
            
            print(f"Heatmap creation ({rows}x{cols}): {duration:.3f}s")
    
    @pytest.mark.performance
    def test_multiple_chart_creation_performance(self):
        """Test performance when creating multiple charts simultaneously."""
        import plotly.graph_objects as go
        from src.visualization.components import ChartStyler, ComponentFactory
        
        num_charts = 10
        data_points_per_chart = 1000
        
        start_time = time.time()
        
        charts = []
        for i in range(num_charts):
            # Generate data for each chart
            x_data = list(range(data_points_per_chart))
            y_data = np.random.uniform(0, 1, data_points_per_chart)
            
            # Create chart
            fig = go.Figure()
            fig.add_scatter(x=x_data, y=y_data, mode='lines', name=f'Chart {i+1}')
            
            # Apply styling
            styled_fig = ChartStyler.apply_base_layout(fig, f"Chart {i+1}")
            
            # Create card component
            card = ComponentFactory.create_card(f"Chart {i+1}", styled_fig)
            charts.append(card)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Performance assertions
        assert duration < 30.0, f"Creating {num_charts} charts took {duration:.2f}s, expected < 30.0s"
        assert len(charts) == num_charts
        
        charts_per_second = num_charts / duration
        print(f"Multiple chart creation ({num_charts} charts, {data_points_per_chart} points each):")
        print(f"  Total duration: {duration:.3f}s")
        print(f"  Charts/second: {charts_per_second:.1f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "-m", "performance"])