"""
Machine Learning Pipeline for Churn Prediction.

This module implements the complete ML pipeline including feature engineering,
model training, evaluation, and prediction scoring for player churn prediction.
"""
import logging
from typing import List, Dict, Tuple, Optional, Any
import pandas as pd
import numpy as np
from datetime import datetime, date
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
import joblib
import os

from ..models.churn_features import ChurnFeatures
from ..models.player_profile import PlayerProfile

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Feature engineering for churn prediction model."""
    
    def __init__(self):
        """Initialize feature engineer."""
        self.feature_columns = [
            'days_since_last_session',
            'sessions_last_7_days', 
            'avg_session_duration_minutes',
            'levels_completed_last_week',
            'purchases_last_30_days',
            'social_connections',
            # Engineered features
            'session_frequency_score',
            'engagement_score',
            'monetary_score',
            'social_score',
            'recency_score'
        ]
    
    def engineer_features(self, churn_features: List[ChurnFeatures]) -> pd.DataFrame:
        """
        Engineer features from raw churn features data.
        
        Args:
            churn_features: List of ChurnFeatures objects
            
        Returns:
            DataFrame with engineered features
        """
        if not churn_features:
            raise ValueError("churn_features cannot be empty")
        
        # Convert to DataFrame
        data = []
        for cf in churn_features:
            data.append({
                'player_id': cf.player_id,
                'days_since_last_session': cf.days_since_last_session,
                'sessions_last_7_days': cf.sessions_last_7_days,
                'avg_session_duration_minutes': cf.avg_session_duration_minutes,
                'levels_completed_last_week': cf.levels_completed_last_week,
                'purchases_last_30_days': cf.purchases_last_30_days,
                'social_connections': cf.social_connections,
                'feature_date': cf.feature_date
            })
        
        df = pd.DataFrame(data)
        
        # Engineer additional features
        df['session_frequency_score'] = self._calculate_session_frequency_score(df)
        df['engagement_score'] = self._calculate_engagement_score(df)
        df['monetary_score'] = self._calculate_monetary_score(df)
        df['social_score'] = self._calculate_social_score(df)
        df['recency_score'] = self._calculate_recency_score(df)
        
        return df
    
    def _calculate_session_frequency_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate session frequency score (0-1 scale)."""
        # Normalize sessions per week to 0-1 scale (assuming max 21 sessions per week)
        max_sessions = 21
        return np.clip(df['sessions_last_7_days'] / max_sessions, 0, 1)
    
    def _calculate_engagement_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate engagement score based on session duration and levels completed."""
        # Normalize session duration (assuming max 120 minutes average)
        duration_score = np.clip(df['avg_session_duration_minutes'] / 120, 0, 1)
        
        # Normalize levels completed (assuming max 50 levels per week)
        levels_score = np.clip(df['levels_completed_last_week'] / 50, 0, 1)
        
        # Weighted combination
        return 0.6 * duration_score + 0.4 * levels_score
    
    def _calculate_monetary_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate monetary score based on recent purchases."""
        # Log transform to handle wide range of purchase amounts
        purchases = df['purchases_last_30_days']
        log_purchases = np.log1p(purchases)  # log(1 + x) to handle zeros
        
        # Normalize to 0-1 scale (assuming max log purchase of 10)
        return np.clip(log_purchases / 10, 0, 1)
    
    def _calculate_social_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate social score based on connections."""
        # Normalize social connections (assuming max 100 connections)
        return np.clip(df['social_connections'] / 100, 0, 1)
    
    def _calculate_recency_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate recency score (higher score = more recent activity)."""
        # Inverse of days since last session, normalized
        days = df['days_since_last_session']
        # Use exponential decay: score = exp(-days/7)
        return np.exp(-days / 7)
    
    def get_feature_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract feature matrix for model training/prediction.
        
        Args:
            df: DataFrame with engineered features
            
        Returns:
            Feature matrix as numpy array
        """
        return df[self.feature_columns].values


class ChurnPredictor:
    """Churn prediction model trainer and predictor."""
    
    def __init__(self):
        """Initialize churn predictor."""
        self.models = {
            'random_forest': RandomForestClassifier(random_state=42),
            'gradient_boosting': GradientBoostingClassifier(random_state=42)
        }
        self.best_model = None
        self.best_model_name = None
        self.scaler = StandardScaler()
        self.feature_engineer = FeatureEngineer()
        
        # Hyperparameter grids
        self.param_grids = {
            'random_forest': {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            },
            'gradient_boosting': {
                'n_estimators': [100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.1, 0.2],
                'min_samples_split': [2, 5]
            }
        }
    
    def prepare_training_data(
        self, 
        churn_features: List[ChurnFeatures], 
        labels: List[bool]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data from churn features and labels.
        
        Args:
            churn_features: List of ChurnFeatures objects
            labels: List of churn labels (True = churned, False = retained)
            
        Returns:
            Tuple of (X, y) arrays
        """
        if len(churn_features) != len(labels):
            raise ValueError("churn_features and labels must have same length")
        
        if len(churn_features) == 0:
            raise ValueError("Training data cannot be empty")
        
        # Engineer features
        df = self.feature_engineer.engineer_features(churn_features)
        X = self.feature_engineer.get_feature_matrix(df)
        y = np.array(labels, dtype=int)
        
        return X, y
    
    def train_models(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        test_size: float = 0.2,
        cv_folds: int = 5
    ) -> Dict[str, Dict[str, Any]]:
        """
        Train and evaluate multiple models with hyperparameter tuning.
        
        Args:
            X: Feature matrix
            y: Target labels
            test_size: Proportion of data for testing
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary with model performance results
        """
        if X.shape[0] != len(y):
            raise ValueError("X and y must have same number of samples")
        
        if X.shape[0] < 10:
            raise ValueError("Need at least 10 samples for training")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        results = {}
        best_score = 0
        
        for model_name, model in self.models.items():
            logger.info(f"Training {model_name}...")
            
            # Hyperparameter tuning
            grid_search = GridSearchCV(
                model, 
                self.param_grids[model_name],
                cv=cv_folds,
                scoring='roc_auc',
                n_jobs=-1
            )
            
            grid_search.fit(X_train_scaled, y_train)
            best_model = grid_search.best_estimator_
            
            # Evaluate on test set
            y_pred = best_model.predict(X_test_scaled)
            y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_pred_proba),
                'best_params': grid_search.best_params_,
                'cv_score': grid_search.best_score_
            }
            
            results[model_name] = {
                'model': best_model,
                'metrics': metrics
            }
            
            # Track best model
            if metrics['roc_auc'] > best_score:
                best_score = metrics['roc_auc']
                self.best_model = best_model
                self.best_model_name = model_name
        
        logger.info(f"Best model: {self.best_model_name} (ROC-AUC: {best_score:.4f})")
        return results
    
    def cross_validate_model(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        cv_folds: int = 5
    ) -> Dict[str, float]:
        """
        Perform cross-validation on the best model.
        
        Args:
            X: Feature matrix
            y: Target labels
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary with cross-validation scores
        """
        if self.best_model is None:
            raise ValueError("No trained model available. Call train_models first.")
        
        X_scaled = self.scaler.transform(X)
        
        # Cross-validation scores
        cv_scores = {
            'accuracy': cross_val_score(self.best_model, X_scaled, y, cv=cv_folds, scoring='accuracy'),
            'precision': cross_val_score(self.best_model, X_scaled, y, cv=cv_folds, scoring='precision'),
            'recall': cross_val_score(self.best_model, X_scaled, y, cv=cv_folds, scoring='recall'),
            'f1': cross_val_score(self.best_model, X_scaled, y, cv=cv_folds, scoring='f1'),
            'roc_auc': cross_val_score(self.best_model, X_scaled, y, cv=cv_folds, scoring='roc_auc')
        }
        
        # Calculate mean and std for each metric
        cv_results = {}
        for metric, scores in cv_scores.items():
            cv_results[f'{metric}_mean'] = scores.mean()
            cv_results[f'{metric}_std'] = scores.std()
        
        return cv_results
    
    def predict_churn_probability(self, churn_features: List[ChurnFeatures]) -> List[float]:
        """
        Predict churn probability for players.
        
        Args:
            churn_features: List of ChurnFeatures objects
            
        Returns:
            List of churn probabilities (0-1)
        """
        if self.best_model is None:
            raise ValueError("No trained model available. Call train_models first.")
        
        if not churn_features:
            return []
        
        # Engineer features
        df = self.feature_engineer.engineer_features(churn_features)
        X = self.feature_engineer.get_feature_matrix(df)
        X_scaled = self.scaler.transform(X)
        
        # Get probabilities
        probabilities = self.best_model.predict_proba(X_scaled)[:, 1]
        return probabilities.tolist()
    
    def predict_churn_binary(self, churn_features: List[ChurnFeatures]) -> List[bool]:
        """
        Predict binary churn labels for players.
        
        Args:
            churn_features: List of ChurnFeatures objects
            
        Returns:
            List of churn predictions (True = will churn, False = will retain)
        """
        if self.best_model is None:
            raise ValueError("No trained model available. Call train_models first.")
        
        if not churn_features:
            return []
        
        # Engineer features
        df = self.feature_engineer.engineer_features(churn_features)
        X = self.feature_engineer.get_feature_matrix(df)
        X_scaled = self.scaler.transform(X)
        
        # Get predictions
        predictions = self.best_model.predict(X_scaled)
        return [bool(pred) for pred in predictions]
    
    def save_model(self, filepath: str) -> None:
        """
        Save trained model and scaler to file.
        
        Args:
            filepath: Path to save model
        """
        if self.best_model is None:
            raise ValueError("No trained model to save")
        
        model_data = {
            'model': self.best_model,
            'scaler': self.scaler,
            'model_name': self.best_model_name,
            'feature_columns': self.feature_engineer.feature_columns
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load trained model and scaler from file.
        
        Args:
            filepath: Path to load model from
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        self.best_model = model_data['model']
        self.scaler = model_data['scaler']
        self.best_model_name = model_data['model_name']
        
        # Verify feature columns match
        if model_data['feature_columns'] != self.feature_engineer.feature_columns:
            logger.warning("Feature columns mismatch between saved model and current feature engineer")
        
        logger.info(f"Model loaded from {filepath}")


class ModelEvaluator:
    """Model evaluation utilities."""
    
    @staticmethod
    def evaluate_model_performance(
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        y_pred_proba: np.ndarray
    ) -> Dict[str, Any]:
        """
        Comprehensive model evaluation.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities
            
        Returns:
            Dictionary with evaluation metrics
        """
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, y_pred_proba),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
            'classification_report': classification_report(y_true, y_pred, output_dict=True)
        }
    
    @staticmethod
    def check_model_accuracy_threshold(
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        threshold: float = 0.8
    ) -> bool:
        """
        Check if model meets accuracy threshold requirement.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            threshold: Minimum accuracy threshold
            
        Returns:
            True if model meets threshold, False otherwise
        """
        accuracy = accuracy_score(y_true, y_pred)
        return accuracy >= threshold