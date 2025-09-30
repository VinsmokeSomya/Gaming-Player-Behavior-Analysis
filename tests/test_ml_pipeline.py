"""
Unit tests for ML Pipeline components.
"""
import pytest
import numpy as np
import pandas as pd
from datetime import date, datetime
from unittest.mock import Mock, patch
import tempfile
import os

from src.services.ml_pipeline import FeatureEngineer, ChurnPredictor, ModelEvaluator
from src.models.churn_features import ChurnFeatures


class TestFeatureEngineer:
    """Test cases for FeatureEngineer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.feature_engineer = FeatureEngineer()
        self.sample_features = [
            ChurnFeatures(
                player_id="player_1",
                days_since_last_session=1,
                sessions_last_7_days=5,
                avg_session_duration_minutes=30.0,
                levels_completed_last_week=10,
                purchases_last_30_days=25.0,
                social_connections=15,
                feature_date=date(2024, 1, 1)
            ),
            ChurnFeatures(
                player_id="player_2", 
                days_since_last_session=7,
                sessions_last_7_days=0,
                avg_session_duration_minutes=5.0,
                levels_completed_last_week=0,
                purchases_last_30_days=0.0,
                social_connections=2,
                feature_date=date(2024, 1, 1)
            )
        ]
    
    def test_engineer_features_success(self):
        """Test successful feature engineering."""
        df = self.feature_engineer.engineer_features(self.sample_features)
        
        # Check DataFrame structure
        assert len(df) == 2
        assert 'player_id' in df.columns
        assert 'session_frequency_score' in df.columns
        assert 'engagement_score' in df.columns
        assert 'monetary_score' in df.columns
        assert 'social_score' in df.columns
        assert 'recency_score' in df.columns
        
        # Check engineered features are numeric
        assert df['session_frequency_score'].dtype in [np.float64, np.float32]
        assert df['engagement_score'].dtype in [np.float64, np.float32]
        assert df['monetary_score'].dtype in [np.float64, np.float32]
        assert df['social_score'].dtype in [np.float64, np.float32]
        assert df['recency_score'].dtype in [np.float64, np.float32]
        
        # Check score ranges (0-1)
        for col in ['session_frequency_score', 'engagement_score', 'monetary_score', 'social_score', 'recency_score']:
            assert df[col].min() >= 0
            assert df[col].max() <= 1
    
    def test_engineer_features_empty_input(self):
        """Test feature engineering with empty input."""
        with pytest.raises(ValueError, match="churn_features cannot be empty"):
            self.feature_engineer.engineer_features([])
    
    def test_session_frequency_score_calculation(self):
        """Test session frequency score calculation."""
        df = pd.DataFrame({
            'sessions_last_7_days': [0, 7, 21, 42]  # Including value above max
        })
        
        scores = self.feature_engineer._calculate_session_frequency_score(df)
        
        assert scores.iloc[0] == 0.0  # 0 sessions
        assert scores.iloc[1] == pytest.approx(7/21, rel=1e-3)  # 7 sessions
        assert scores.iloc[2] == 1.0  # 21 sessions (max)
        assert scores.iloc[3] == 1.0  # 42 sessions (clipped to 1.0)
    
    def test_engagement_score_calculation(self):
        """Test engagement score calculation."""
        df = pd.DataFrame({
            'avg_session_duration_minutes': [0, 60, 120],
            'levels_completed_last_week': [0, 25, 50]
        })
        
        scores = self.feature_engineer._calculate_engagement_score(df)
        
        # Check that scores are in valid range
        assert all(0 <= score <= 1 for score in scores)
        
        # Check that higher values produce higher scores
        assert scores.iloc[0] < scores.iloc[1] < scores.iloc[2]
    
    def test_monetary_score_calculation(self):
        """Test monetary score calculation."""
        df = pd.DataFrame({
            'purchases_last_30_days': [0, 1, 10, 100]
        })
        
        scores = self.feature_engineer._calculate_monetary_score(df)
        
        # Check that scores are in valid range
        assert all(0 <= score <= 1 for score in scores)
        
        # Check that higher purchases produce higher scores
        assert scores.iloc[0] < scores.iloc[1] < scores.iloc[2] < scores.iloc[3]
    
    def test_social_score_calculation(self):
        """Test social score calculation."""
        df = pd.DataFrame({
            'social_connections': [0, 25, 100, 200]  # Including value above max
        })
        
        scores = self.feature_engineer._calculate_social_score(df)
        
        assert scores.iloc[0] == 0.0  # 0 connections
        assert scores.iloc[1] == 0.25  # 25 connections
        assert scores.iloc[2] == 1.0  # 100 connections (max)
        assert scores.iloc[3] == 1.0  # 200 connections (clipped to 1.0)
    
    def test_recency_score_calculation(self):
        """Test recency score calculation."""
        df = pd.DataFrame({
            'days_since_last_session': [0, 7, 14]
        })
        
        scores = self.feature_engineer._calculate_recency_score(df)
        
        # Check that scores are in valid range
        assert all(0 <= score <= 1 for score in scores)
        
        # Check that more recent activity produces higher scores
        assert scores.iloc[0] > scores.iloc[1] > scores.iloc[2]
        
        # Check specific values
        assert scores.iloc[0] == 1.0  # 0 days = perfect recency
        assert scores.iloc[1] == pytest.approx(np.exp(-1), rel=1e-3)  # 7 days
    
    def test_get_feature_matrix(self):
        """Test feature matrix extraction."""
        df = self.feature_engineer.engineer_features(self.sample_features)
        X = self.feature_engineer.get_feature_matrix(df)
        
        # Check matrix shape
        assert X.shape == (2, len(self.feature_engineer.feature_columns))
        
        # Check that it's a numpy array
        assert isinstance(X, np.ndarray)
        
        # Check that all values are numeric
        assert np.isfinite(X).all()


class TestChurnPredictor:
    """Test cases for ChurnPredictor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.predictor = ChurnPredictor()
        self.sample_features = [
            ChurnFeatures(
                player_id=f"player_{i}",
                days_since_last_session=i % 8,  # Keep within 7 days to avoid validation issues
                sessions_last_7_days=max(0, 7 - i % 8) if i % 8 <= 7 else 0,  # Ensure logical consistency
                avg_session_duration_minutes=max(5.0, 30.0 - i * 2),  # Ensure positive duration
                levels_completed_last_week=max(0, 10 - i),
                purchases_last_30_days=max(0, 25.0 - i * 2),
                social_connections=max(0, 15 - i),
                feature_date=date(2024, 1, 1)
            )
            for i in range(20)
        ]
        # Create labels: players with high days_since_last_session are more likely to churn
        self.sample_labels = [features.days_since_last_session > 4 for features in self.sample_features]
    
    def test_prepare_training_data_success(self):
        """Test successful training data preparation."""
        X, y = self.predictor.prepare_training_data(self.sample_features, self.sample_labels)
        
        # Check shapes
        assert X.shape[0] == len(self.sample_features)
        assert len(y) == len(self.sample_labels)
        assert X.shape[1] == len(self.predictor.feature_engineer.feature_columns)
        
        # Check data types
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert y.dtype == int
    
    def test_prepare_training_data_length_mismatch(self):
        """Test training data preparation with mismatched lengths."""
        with pytest.raises(ValueError, match="churn_features and labels must have same length"):
            self.predictor.prepare_training_data(self.sample_features, self.sample_labels[:-1])
    
    def test_prepare_training_data_empty(self):
        """Test training data preparation with empty data."""
        with pytest.raises(ValueError, match="Training data cannot be empty"):
            self.predictor.prepare_training_data([], [])
    
    def test_train_models_success(self):
        """Test successful model training."""
        X, y = self.predictor.prepare_training_data(self.sample_features, self.sample_labels)
        
        # Mock GridSearchCV to speed up tests
        with patch('src.services.ml_pipeline.GridSearchCV') as mock_grid:
            # Create mock best estimator
            mock_estimator = Mock()
            mock_estimator.predict.return_value = np.array([0, 1] * 2)  # Alternating predictions
            mock_estimator.predict_proba.return_value = np.array([[0.8, 0.2], [0.3, 0.7]] * 2)
            
            mock_grid.return_value.best_estimator_ = mock_estimator
            mock_grid.return_value.best_params_ = {'n_estimators': 100}
            mock_grid.return_value.best_score_ = 0.85
            mock_grid.return_value.fit.return_value = None
            
            results = self.predictor.train_models(X, y, test_size=0.2, cv_folds=3)
        
        # Check results structure
        assert 'random_forest' in results
        assert 'gradient_boosting' in results
        
        for model_name in results:
            assert 'model' in results[model_name]
            assert 'metrics' in results[model_name]
            
            metrics = results[model_name]['metrics']
            assert 'accuracy' in metrics
            assert 'precision' in metrics
            assert 'recall' in metrics
            assert 'f1' in metrics
            assert 'roc_auc' in metrics
            assert 'best_params' in metrics
            assert 'cv_score' in metrics
        
        # Check that best model is set
        assert self.predictor.best_model is not None
        assert self.predictor.best_model_name is not None
    
    def test_train_models_insufficient_data(self):
        """Test model training with insufficient data."""
        small_features = self.sample_features[:5]
        small_labels = self.sample_labels[:5]
        X, y = self.predictor.prepare_training_data(small_features, small_labels)
        
        with pytest.raises(ValueError, match="Need at least 10 samples for training"):
            self.predictor.train_models(X, y)
    
    def test_train_models_shape_mismatch(self):
        """Test model training with shape mismatch."""
        X = np.random.rand(10, 5)
        y = np.array([0, 1] * 4)  # Only 8 labels for 10 samples
        
        with pytest.raises(ValueError, match="X and y must have same number of samples"):
            self.predictor.train_models(X, y)
    
    def test_cross_validate_model_no_trained_model(self):
        """Test cross-validation without trained model."""
        X = np.random.rand(10, 5)
        y = np.array([0, 1] * 5)
        
        with pytest.raises(ValueError, match="No trained model available"):
            self.predictor.cross_validate_model(X, y)
    
    def test_predict_churn_probability_no_model(self):
        """Test prediction without trained model."""
        with pytest.raises(ValueError, match="No trained model available"):
            self.predictor.predict_churn_probability(self.sample_features)
    
    def test_predict_churn_probability_empty_input(self):
        """Test prediction with empty input."""
        # Set up a mock trained model
        self.predictor.best_model = Mock()
        result = self.predictor.predict_churn_probability([])
        assert result == []
    
    def test_predict_churn_binary_no_model(self):
        """Test binary prediction without trained model."""
        with pytest.raises(ValueError, match="No trained model available"):
            self.predictor.predict_churn_binary(self.sample_features)
    
    def test_predict_churn_binary_empty_input(self):
        """Test binary prediction with empty input."""
        # Set up a mock trained model
        self.predictor.best_model = Mock()
        result = self.predictor.predict_churn_binary([])
        assert result == []
    
    def test_save_model_no_trained_model(self):
        """Test saving model without trained model."""
        with tempfile.NamedTemporaryFile() as tmp:
            with pytest.raises(ValueError, match="No trained model to save"):
                self.predictor.save_model(tmp.name)
    
    def test_save_and_load_model_success(self):
        """Test successful model saving and loading."""
        # Use a real sklearn model instead of Mock to avoid pickling issues
        from sklearn.ensemble import RandomForestClassifier
        real_model = RandomForestClassifier(n_estimators=10, random_state=42)
        
        # Train on dummy data to make it a valid model
        X_dummy = np.random.rand(10, 5)
        y_dummy = np.random.choice([0, 1], size=10)
        real_model.fit(X_dummy, y_dummy)
        
        self.predictor.best_model = real_model
        self.predictor.best_model_name = "test_model"
        
        # Use a temporary directory instead of NamedTemporaryFile to avoid Windows issues
        import tempfile
        import shutil
        temp_dir = tempfile.mkdtemp()
        filepath = os.path.join(temp_dir, 'test_model.joblib')
        
        try:
            # Save model
            self.predictor.save_model(filepath)
            
            # Create new predictor and load model
            new_predictor = ChurnPredictor()
            new_predictor.load_model(filepath)
            
            # Check that model was loaded
            assert new_predictor.best_model is not None
            assert new_predictor.best_model_name == "test_model"
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_load_model_file_not_found(self):
        """Test loading model from non-existent file."""
        with pytest.raises(FileNotFoundError, match="Model file not found"):
            self.predictor.load_model("non_existent_file.joblib")


class TestModelEvaluator:
    """Test cases for ModelEvaluator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0])
        self.y_pred = np.array([0, 1, 1, 1, 0, 0, 1, 0])
        self.y_pred_proba = np.array([0.1, 0.6, 0.8, 0.9, 0.2, 0.4, 0.7, 0.3])
    
    def test_evaluate_model_performance(self):
        """Test comprehensive model evaluation."""
        results = ModelEvaluator.evaluate_model_performance(
            self.y_true, self.y_pred, self.y_pred_proba
        )
        
        # Check that all expected metrics are present
        expected_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 
                          'confusion_matrix', 'classification_report']
        for metric in expected_metrics:
            assert metric in results
        
        # Check that metrics are numeric (except confusion_matrix and classification_report)
        numeric_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        for metric in numeric_metrics:
            assert isinstance(results[metric], (int, float))
            assert 0 <= results[metric] <= 1
        
        # Check confusion matrix structure
        cm = results['confusion_matrix']
        assert len(cm) == 2  # Binary classification
        assert len(cm[0]) == 2
        assert len(cm[1]) == 2
        
        # Check classification report structure
        cr = results['classification_report']
        assert isinstance(cr, dict)
        assert 'accuracy' in cr
    
    def test_check_model_accuracy_threshold_pass(self):
        """Test accuracy threshold check - passing case."""
        # Create predictions with high accuracy
        y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0])
        y_pred = np.array([0, 0, 1, 1, 0, 1, 1, 0])  # Perfect predictions
        
        result = ModelEvaluator.check_model_accuracy_threshold(y_true, y_pred, threshold=0.8)
        assert result is True
    
    def test_check_model_accuracy_threshold_fail(self):
        """Test accuracy threshold check - failing case."""
        # Create predictions with low accuracy
        y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0])
        y_pred = np.array([1, 1, 0, 0, 1, 0, 0, 1])  # All wrong predictions
        
        result = ModelEvaluator.check_model_accuracy_threshold(y_true, y_pred, threshold=0.8)
        assert result is False
    
    def test_check_model_accuracy_threshold_edge_case(self):
        """Test accuracy threshold check - edge case."""
        # Create predictions with exactly threshold accuracy
        y_true = np.array([0, 0, 1, 1, 0])
        y_pred = np.array([0, 0, 1, 1, 1])  # 4/5 = 0.8 accuracy
        
        result = ModelEvaluator.check_model_accuracy_threshold(y_true, y_pred, threshold=0.8)
        assert result is True


# Integration tests
class TestMLPipelineIntegration:
    """Integration tests for ML pipeline components."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.sample_features = [
            ChurnFeatures(
                player_id=f"player_{i}",
                days_since_last_session=i % 8,  # Keep within 7 days to avoid validation issues
                sessions_last_7_days=max(0, 7 - i % 8) if i % 8 <= 7 else 0,  # Ensure logical consistency
                avg_session_duration_minutes=max(5.0, 30.0 - i * 2),
                levels_completed_last_week=max(0, 10 - i),
                purchases_last_30_days=max(0, 25.0 - i * 2),
                social_connections=max(0, 15 - i),
                feature_date=date(2024, 1, 1)
            )
            for i in range(50)  # Larger dataset for integration test
        ]
        # Create realistic labels based on multiple factors
        self.sample_labels = []
        for features in self.sample_features:
            # High churn risk if multiple negative indicators
            risk_score = 0
            if features.days_since_last_session > 5:
                risk_score += 2
            if features.sessions_last_7_days == 0:
                risk_score += 2
            if features.avg_session_duration_minutes < 10:
                risk_score += 1
            if features.purchases_last_30_days == 0:
                risk_score += 1
            
            self.sample_labels.append(risk_score >= 3)
    
    def test_end_to_end_pipeline(self):
        """Test complete ML pipeline from features to predictions."""
        predictor = ChurnPredictor()
        
        # Prepare training data
        X, y = predictor.prepare_training_data(self.sample_features, self.sample_labels)
        
        # Mock the training to speed up test
        with patch('src.services.ml_pipeline.GridSearchCV') as mock_grid:
            # Create mock model that makes reasonable predictions
            mock_model = Mock()
            mock_model.predict.return_value = np.random.choice([0, 1], size=len(y)//5)
            mock_model.predict_proba.return_value = np.random.rand(len(y)//5, 2)
            
            mock_grid.return_value.best_estimator_ = mock_model
            mock_grid.return_value.best_params_ = {'n_estimators': 100}
            mock_grid.return_value.best_score_ = 0.85
            
            # Train models
            results = predictor.train_models(X, y)
        
        # Test predictions
        test_features = self.sample_features[:5]
        
        # Mock the scaler and model for predictions
        predictor.scaler.transform = Mock(return_value=np.random.rand(5, len(predictor.feature_engineer.feature_columns)))
        predictor.best_model.predict_proba = Mock(return_value=np.random.rand(5, 2))
        predictor.best_model.predict = Mock(return_value=np.random.choice([0, 1], size=5))
        
        probabilities = predictor.predict_churn_probability(test_features)
        binary_predictions = predictor.predict_churn_binary(test_features)
        
        # Verify results
        assert len(probabilities) == 5
        assert len(binary_predictions) == 5
        assert all(0 <= prob <= 1 for prob in probabilities)
        assert all(isinstance(pred, bool) for pred in binary_predictions)
    
    def test_feature_engineering_consistency(self):
        """Test that feature engineering produces consistent results."""
        engineer = FeatureEngineer()
        
        # Engineer features twice
        df1 = engineer.engineer_features(self.sample_features)
        df2 = engineer.engineer_features(self.sample_features)
        
        # Results should be identical
        pd.testing.assert_frame_equal(df1, df2)
        
        # Feature matrix should be consistent
        X1 = engineer.get_feature_matrix(df1)
        X2 = engineer.get_feature_matrix(df2)
        
        np.testing.assert_array_equal(X1, X2)