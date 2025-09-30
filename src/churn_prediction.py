"""
Churn Prediction Model for Gaming Players
Target: Achieve 80% accuracy using scikit-learn
"""
import pandas as pd
import numpy as np
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

class ChurnPredictor:
    def __init__(self, db_path):
        self.db_path = db_path
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        
    def load_data(self):
        """Load and prepare data for churn prediction"""
        conn = sqlite3.connect(self.db_path)
        
        # Get player features
        query = """
        SELECT 
            p.player_id,
            p.country,
            p.platform,
            p.age_group,
            julianday('now') - julianday(p.registration_date) as days_since_registration,
            ltv.total_sessions,
            ltv.total_playtime,
            ltv.total_purchases,
            ltv.total_revenue,
            ltv.avg_session_length,
            ltv.max_level,
            ltv.lifetime_days,
            cp.days_since_last_activity,
            cp.is_churned
        FROM players p
        JOIN player_ltv ltv ON p.player_id = ltv.player_id
        JOIN churned_players cp ON p.player_id = cp.player_id
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        return df
    
    def engineer_features(self, df):
        """Create additional features for better prediction"""
        # Engagement metrics
        df['sessions_per_day'] = df['total_sessions'] / df['lifetime_days']
        df['revenue_per_session'] = df['total_revenue'] / df['total_sessions']
        df['playtime_per_session'] = df['total_playtime'] / df['total_sessions']
        
        # Behavioral patterns
        df['is_spender'] = (df['total_revenue'] > 0).astype(int)
        df['high_engagement'] = (df['sessions_per_day'] > df['sessions_per_day'].median()).astype(int)
        df['level_progression_rate'] = df['max_level'] / df['lifetime_days']
        
        # Replace infinite values
        df = df.replace([np.inf, -np.inf], 0)
        df = df.fillna(0)
        
        return df
    
    def prepare_features(self, df):
        """Prepare features for machine learning"""
        # Categorical features to encode
        categorical_features = ['country', 'platform', 'age_group']
        
        # Encode categorical variables
        for feature in categorical_features:
            if feature not in self.label_encoders:
                self.label_encoders[feature] = LabelEncoder()
                df[feature + '_encoded'] = self.label_encoders[feature].fit_transform(df[feature])
            else:
                df[feature + '_encoded'] = self.label_encoders[feature].transform(df[feature])
        
        # Select features for modeling
        feature_columns = [
            'days_since_registration', 'total_sessions', 'total_playtime',
            'total_purchases', 'total_revenue', 'avg_session_length',
            'max_level', 'lifetime_days', 'sessions_per_day',
            'revenue_per_session', 'playtime_per_session', 'is_spender',
            'high_engagement', 'level_progression_rate',
            'country_encoded', 'platform_encoded', 'age_group_encoded'
        ]
        
        X = df[feature_columns]
        y = df['is_churned']
        
        self.feature_names = feature_columns
        
        return X, y
    
    def train_model(self, X, y):
        """Train the churn prediction model"""
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Try multiple models to achieve 80% accuracy
        models = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000)
        }
        
        best_accuracy = 0
        best_model = None
        best_model_name = None
        
        print("Training multiple models...")
        for name, model in models.items():
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            print(f"{name} Accuracy: {accuracy:.4f}")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model
                best_model_name = name
        
        self.model = best_model
        
        # Final evaluation
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\nBest Model: {best_model_name}")
        print(f"Final Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Feature importance (if Random Forest)
        if best_model_name == 'RandomForest':
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\nTop 10 Most Important Features:")
            print(feature_importance.head(10))
        
        return accuracy, y_test, y_pred
    
    def save_model(self, model_path='models'):
        """Save the trained model and preprocessors"""
        os.makedirs(model_path, exist_ok=True)
        
        joblib.dump(self.model, f'{model_path}/churn_model.pkl')
        joblib.dump(self.scaler, f'{model_path}/scaler.pkl')
        joblib.dump(self.label_encoders, f'{model_path}/label_encoders.pkl')
        
        print(f"Model saved to {model_path}/")
    
    def predict_churn(self, player_data):
        """Predict churn for new player data"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Engineer features
        player_data = self.engineer_features(player_data)
        
        # Prepare features
        X, _ = self.prepare_features(player_data)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Predict
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)[:, 1]
        
        return predictions, probabilities

def main():
    """Main function to run churn prediction analysis"""
    db_path = '/home/runner/work/Gaming-Player-Behavior-Analysis/Gaming-Player-Behavior-Analysis/data/gaming_data.db'
    model_path = '/home/runner/work/Gaming-Player-Behavior-Analysis/Gaming-Player-Behavior-Analysis/models'
    
    # Initialize predictor
    predictor = ChurnPredictor(db_path)
    
    # Load and prepare data
    print("Loading data...")
    df = predictor.load_data()
    df = predictor.engineer_features(df)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Churn rate: {df['is_churned'].mean():.2%}")
    
    # Prepare features
    X, y = predictor.prepare_features(df)
    
    # Train model
    accuracy, y_test, y_pred = predictor.train_model(X, y)
    
    # Save model
    predictor.save_model(model_path)
    
    print(f"\nTarget achieved: {accuracy >= 0.80}")
    
    return predictor, accuracy

if __name__ == "__main__":
    main()