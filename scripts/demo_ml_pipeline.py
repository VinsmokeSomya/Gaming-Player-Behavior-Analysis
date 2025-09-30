#!/usr/bin/env python3
"""
Demo script for the ML Pipeline - Churn Prediction.

This script demonstrates how to use the churn prediction ML pipeline
including feature engineering, model training, and prediction.
"""
import sys
import os
from datetime import date
import numpy as np

# Add project root to path for imports
project_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, project_root)

from src.services.ml_pipeline import FeatureEngineer, ChurnPredictor, ModelEvaluator
from src.models.churn_features import ChurnFeatures


def generate_sample_data(n_samples=100):
    """Generate sample churn features data for demonstration."""
    features = []
    labels = []
    
    for i in range(n_samples):
        # Create realistic player data
        days_since_last = np.random.randint(0, 14)
        sessions_last_7 = np.random.randint(0, 8) if days_since_last <= 7 else 0
        avg_duration = max(5.0, np.random.normal(25, 10))
        levels_completed = max(0, np.random.randint(0, 20))
        purchases = max(0, np.random.exponential(10))
        social_connections = max(0, np.random.randint(0, 50))
        
        feature = ChurnFeatures(
            player_id=f"demo_player_{i}",
            days_since_last_session=days_since_last,
            sessions_last_7_days=sessions_last_7,
            avg_session_duration_minutes=avg_duration,
            levels_completed_last_week=levels_completed,
            purchases_last_30_days=purchases,
            social_connections=social_connections,
            feature_date=date(2024, 1, 1)
        )
        
        # Create realistic churn labels based on features
        churn_risk = 0
        if days_since_last > 7:
            churn_risk += 3
        if sessions_last_7 == 0:
            churn_risk += 2
        if avg_duration < 10:
            churn_risk += 1
        if purchases == 0:
            churn_risk += 1
        
        # Add some randomness
        churn_risk += np.random.randint(-1, 2)
        
        features.append(feature)
        labels.append(churn_risk >= 3)
    
    return features, labels


def main():
    """Main demonstration function."""
    print("🎮 Churn Prediction ML Pipeline Demo")
    print("=" * 50)
    
    # Generate sample data
    print("\n1. Generating sample player data...")
    features, labels = generate_sample_data(200)
    print(f"   Generated {len(features)} player records")
    print(f"   Churn rate: {sum(labels)/len(labels):.2%}")
    
    # Initialize components
    print("\n2. Initializing ML pipeline components...")
    feature_engineer = FeatureEngineer()
    predictor = ChurnPredictor()
    
    # Feature engineering
    print("\n3. Engineering features...")
    df = feature_engineer.engineer_features(features)
    print(f"   Engineered {len(feature_engineer.feature_columns)} features")
    print(f"   Feature columns: {feature_engineer.feature_columns}")
    
    # Prepare training data
    print("\n4. Preparing training data...")
    X, y = predictor.prepare_training_data(features, labels)
    print(f"   Training data shape: {X.shape}")
    print(f"   Target distribution: {np.bincount(y)}")
    
    # Train models
    print("\n5. Training models (this may take a moment)...")
    results = predictor.train_models(X, y, test_size=0.3, cv_folds=3)
    
    print(f"   Best model: {predictor.best_model_name}")
    for model_name, result in results.items():
        metrics = result['metrics']
        print(f"   {model_name}:")
        print(f"     - Accuracy: {metrics['accuracy']:.3f}")
        print(f"     - Precision: {metrics['precision']:.3f}")
        print(f"     - Recall: {metrics['recall']:.3f}")
        print(f"     - F1-Score: {metrics['f1']:.3f}")
        print(f"     - ROC-AUC: {metrics['roc_auc']:.3f}")
    
    # Cross-validation
    print("\n6. Performing cross-validation...")
    cv_results = predictor.cross_validate_model(X, y, cv_folds=5)
    print("   Cross-validation results:")
    for metric, value in cv_results.items():
        print(f"     - {metric}: {value:.3f}")
    
    # Make predictions on new data
    print("\n7. Making predictions on new players...")
    test_features = features[:10]  # Use first 10 as test
    
    probabilities = predictor.predict_churn_probability(test_features)
    binary_predictions = predictor.predict_churn_binary(test_features)
    
    print("   Prediction results:")
    for i, (feature, prob, pred) in enumerate(zip(test_features, probabilities, binary_predictions)):
        print(f"     Player {feature.player_id}: {prob:.3f} probability, "
              f"{'WILL CHURN' if pred else 'WILL RETAIN'}")
    
    # Check accuracy requirement
    print("\n8. Checking accuracy requirements...")
    # Use the best model to make predictions on test set
    test_size = int(0.3 * len(X))
    X_test = X[-test_size:]
    y_test = y[-test_size:]
    
    # Scale test data
    X_test_scaled = predictor.scaler.transform(X_test)
    y_pred_test = predictor.best_model.predict(X_test_scaled)
    y_pred_proba_test = predictor.best_model.predict_proba(X_test_scaled)[:, 1]
    
    # Evaluate performance
    evaluation = ModelEvaluator.evaluate_model_performance(y_test, y_pred_test, y_pred_proba_test)
    meets_threshold = ModelEvaluator.check_model_accuracy_threshold(y_test, y_pred_test, threshold=0.8)
    
    print(f"   Final test accuracy: {evaluation['accuracy']:.3f}")
    print(f"   Meets 80% accuracy requirement: {'✅ YES' if meets_threshold else '❌ NO'}")
    
    # Save model
    print("\n9. Saving trained model...")
    model_path = "models/churn_predictor.joblib"
    os.makedirs("models", exist_ok=True)
    predictor.save_model(model_path)
    print(f"   Model saved to: {model_path}")
    
    print("\n🎉 Demo completed successfully!")
    print("\nKey achievements:")
    print(f"   ✅ Trained {len(results)} ML models")
    print(f"   ✅ Best model ROC-AUC: {results[predictor.best_model_name]['metrics']['roc_auc']:.3f}")
    print(f"   ✅ Model {'meets' if meets_threshold else 'does not meet'} accuracy requirement")
    print(f"   ✅ Generated probability scores for churn prediction")
    print(f"   ✅ Model saved for future use")


if __name__ == "__main__":
    main()