"""
Data adapter to transform the real gaming dataset into our analytics models.
Converts the online_gaming_behavior_dataset.csv into PlayerProfile, events, and ChurnFeatures.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from typing import List, Dict, Any, Tuple
import random
from dataclasses import asdict
import json
import os

from models import PlayerProfile, ChurnFeatures


class RealDataAdapter:
    """Adapts real gaming dataset to our analytics models."""
    
    def __init__(self, csv_path: str, seed: int = 42):
        """Initialize the adapter with the CSV dataset."""
        self.csv_path = csv_path
        random.seed(seed)
        np.random.seed(seed)
        
        # Load the dataset
        print(f"Loading dataset from {csv_path}...")
        self.df = pd.read_csv(csv_path)
        print(f"Loaded {len(self.df)} players")
        
        # Set up time ranges for realistic dates
        self.end_date = datetime.now() - timedelta(days=1)
        self.start_date = self.end_date - timedelta(days=180)  # 6 months of data
    
    def generate_registration_dates(self) -> pd.Series:
        """Generate realistic registration dates based on player engagement."""
        registration_dates = []
        
        for _, player in self.df.iterrows():
            # More engaged players likely registered earlier
            if player['EngagementLevel'] == 'High':
                days_ago = random.randint(30, 180)  # 1-6 months ago
            elif player['EngagementLevel'] == 'Medium':
                days_ago = random.randint(14, 120)  # 2 weeks - 4 months ago
            else:  # Low engagement
                days_ago = random.randint(7, 60)   # 1 week - 2 months ago
            
            reg_date = self.end_date - timedelta(days=days_ago)
            registration_dates.append(reg_date)
        
        return pd.Series(registration_dates)
    
    def generate_last_active_dates(self, registration_dates: pd.Series) -> pd.Series:
        """Generate last active dates based on engagement level."""
        last_active_dates = []
        
        for i, (_, player) in enumerate(self.df.iterrows()):
            reg_date = registration_dates.iloc[i]
            
            # Calculate days since registration
            days_since_reg = (self.end_date - reg_date).days
            
            # Determine activity based on engagement level
            if player['EngagementLevel'] == 'High':
                # High engagement: active recently
                days_inactive = random.randint(0, 3)
            elif player['EngagementLevel'] == 'Medium':
                # Medium engagement: somewhat active
                days_inactive = random.randint(1, 14)
            else:  # Low engagement
                # Low engagement: might be churned
                days_inactive = random.randint(7, min(60, days_since_reg))
            
            last_active = self.end_date - timedelta(days=days_inactive)
            # Ensure last active is not before registration
            last_active = max(last_active, reg_date)
            last_active_dates.append(last_active)
        
        return pd.Series(last_active_dates)
    
    def calculate_total_sessions(self) -> pd.Series:
        """Calculate total sessions based on sessions per week and time active."""
        total_sessions = []
        
        for _, player in self.df.iterrows():
            sessions_per_week = player['SessionsPerWeek']
            
            # Estimate weeks active (with some variation)
            base_weeks = random.uniform(4, 24)  # 1-6 months
            
            # Adjust based on engagement
            if player['EngagementLevel'] == 'High':
                weeks_multiplier = random.uniform(1.2, 2.0)
            elif player['EngagementLevel'] == 'Medium':
                weeks_multiplier = random.uniform(0.8, 1.5)
            else:
                weeks_multiplier = random.uniform(0.3, 1.0)
            
            weeks_active = base_weeks * weeks_multiplier
            total = int(sessions_per_week * weeks_active)
            total_sessions.append(max(1, int(total)))  # At least 1 session, ensure int
        
        return pd.Series(total_sessions)
    
    def calculate_total_playtime(self, total_sessions: pd.Series) -> pd.Series:
        """Calculate total playtime from sessions and average duration."""
        total_playtime = []
        
        for i, (_, player) in enumerate(self.df.iterrows()):
            avg_duration = player['AvgSessionDurationMinutes']
            sessions = total_sessions.iloc[i]
            
            # Add some variation to make it realistic
            variation = random.uniform(0.7, 1.3)
            playtime = int(sessions * avg_duration * variation)
            total_playtime.append(max(0, int(playtime)))  # Ensure non-negative int
        
        return pd.Series(total_playtime)
    
    def calculate_purchases(self) -> pd.Series:
        """Calculate total purchase amounts based on InGamePurchases flag."""
        purchase_amounts = []
        
        for _, player in self.df.iterrows():
            if player['InGamePurchases'] == 1:
                # Player makes purchases - generate realistic amounts
                base_amount = random.uniform(5, 200)
                
                # Adjust based on engagement and level
                level_multiplier = 1 + (player['PlayerLevel'] / 100)
                
                if player['EngagementLevel'] == 'High':
                    engagement_multiplier = random.uniform(1.5, 3.0)
                elif player['EngagementLevel'] == 'Medium':
                    engagement_multiplier = random.uniform(0.8, 2.0)
                else:
                    engagement_multiplier = random.uniform(0.2, 1.0)
                
                total_amount = base_amount * level_multiplier * engagement_multiplier
                purchase_amounts.append(round(total_amount, 2))
            else:
                purchase_amounts.append(0.0)
        
        return pd.Series(purchase_amounts)
    
    def calculate_churn_risk(self, last_active_dates: pd.Series, 
                           total_sessions: pd.Series, purchase_amounts: pd.Series) -> pd.Series:
        """Calculate churn risk scores based on multiple factors."""
        churn_scores = []
        
        for i, (_, player) in enumerate(self.df.iterrows()):
            last_active = last_active_dates.iloc[i]
            sessions = total_sessions.iloc[i]
            purchases = purchase_amounts.iloc[i]
            
            # Days since last active
            days_inactive = (datetime.now() - last_active).days
            inactivity_risk = min(days_inactive / 30.0, 1.0)
            
            # Session frequency risk
            sessions_per_week = player['SessionsPerWeek']
            session_risk = max(0, 1 - (sessions_per_week / 15.0))
            
            # Purchase behavior risk
            purchase_risk = 1.0 if purchases == 0 else max(0, 1 - (purchases / 100.0))
            
            # Engagement level risk
            engagement_risk = {
                'High': 0.1,
                'Medium': 0.4,
                'Low': 0.8
            }.get(player['EngagementLevel'], 0.5)
            
            # Level progression risk (low level = higher risk)
            level_risk = max(0, 1 - (player['PlayerLevel'] / 100.0))
            
            # Combine all risk factors
            combined_risk = (
                inactivity_risk * 0.3 +
                session_risk * 0.2 +
                purchase_risk * 0.2 +
                engagement_risk * 0.2 +
                level_risk * 0.1
            )
            
            churn_scores.append(min(combined_risk, 1.0))
        
        return pd.Series(churn_scores)
    
    def create_player_profiles(self) -> List[PlayerProfile]:
        """Create PlayerProfile objects from the dataset."""
        print("Creating player profiles...")
        
        # Generate derived fields
        registration_dates = self.generate_registration_dates()
        last_active_dates = self.generate_last_active_dates(registration_dates)
        total_sessions = self.calculate_total_sessions()
        total_playtime = self.calculate_total_playtime(total_sessions)
        purchase_amounts = self.calculate_purchases()
        churn_scores = self.calculate_churn_risk(last_active_dates, total_sessions, purchase_amounts)
        
        profiles = []
        
        for i, (_, player) in enumerate(self.df.iterrows()):
            profile = PlayerProfile(
                player_id=f"player_{int(player['PlayerID'])}",
                registration_date=registration_dates.iloc[i],
                last_active_date=last_active_dates.iloc[i],
                total_sessions=int(total_sessions.iloc[i]),
                total_playtime_minutes=int(total_playtime.iloc[i]),
                highest_level_reached=int(player['PlayerLevel']),
                total_purchases=float(purchase_amounts.iloc[i]),
                churn_risk_score=float(churn_scores.iloc[i]),
                churn_prediction_date=datetime.now()
            )
            profiles.append(profile)
        
        return profiles
    
    def create_churn_features(self, profiles: List[PlayerProfile]) -> List[ChurnFeatures]:
        """Create ChurnFeatures objects based on player profiles and original data."""
        print("Creating churn features...")
        
        churn_features = []
        
        for i, profile in enumerate(profiles):
            player_data = self.df.iloc[i]
            
            # Days since last session
            days_since_last = max(0, (datetime.now() - profile.last_active_date).days)
            
            # Estimate recent sessions based on engagement
            if player_data['EngagementLevel'] == 'High' and days_since_last <= 7:
                max_sessions = max(3, min(7, player_data['SessionsPerWeek']))
                recent_sessions = random.randint(3, max_sessions)
            elif player_data['EngagementLevel'] == 'Medium' and days_since_last <= 7:
                max_sessions = max(1, min(5, player_data['SessionsPerWeek']))
                recent_sessions = random.randint(1, max_sessions)
            elif days_since_last <= 7:
                max_sessions = max(0, min(2, player_data['SessionsPerWeek']))
                recent_sessions = random.randint(0, max_sessions)
            else:
                recent_sessions = 0
            
            # Recent level completions
            if recent_sessions > 0:
                levels_completed = random.randint(0, min(3, recent_sessions))
            else:
                levels_completed = 0
            
            # Recent purchases
            recent_purchases = 0.0
            if profile.total_purchases > 0 and days_since_last <= 30:
                # Estimate recent purchase activity
                monthly_purchase_rate = profile.total_purchases / 6  # Assume 6 months of data
                recent_purchases = random.uniform(0, monthly_purchase_rate * 2)
                recent_purchases = round(recent_purchases, 2)
            
            # Social connections (simulated based on engagement)
            social_connections = {
                'High': random.randint(20, 50),
                'Medium': random.randint(5, 25),
                'Low': random.randint(0, 10)
            }.get(player_data['EngagementLevel'], random.randint(0, 15))
            
            features = ChurnFeatures(
                player_id=profile.player_id,
                days_since_last_session=days_since_last,
                sessions_last_7_days=recent_sessions,
                avg_session_duration_minutes=float(player_data['AvgSessionDurationMinutes']),
                levels_completed_last_week=levels_completed,
                purchases_last_30_days=recent_purchases,
                social_connections=social_connections,
                feature_date=datetime.now().date()
            )
            churn_features.append(features)
        
        return churn_features
    
    def create_synthetic_events(self, profiles: List[PlayerProfile]) -> List[Dict[str, Any]]:
        """Create synthetic events based on player profiles and behavior patterns."""
        print("Creating synthetic events...")
        
        all_events = []
        
        for i, profile in enumerate(profiles):
            player_data = self.df.iloc[i]
            events = []
            
            # Generate session events
            sessions_to_generate = min(profile.total_sessions, 100)  # Limit for performance
            
            current_date = profile.registration_date
            end_date = profile.last_active_date
            
            session_count = 0
            while current_date <= end_date and session_count < sessions_to_generate:
                # Session start
                session_start = current_date + timedelta(
                    hours=random.randint(8, 23),
                    minutes=random.randint(0, 59)
                )
                
                duration = random.randint(
                    max(10, player_data['AvgSessionDurationMinutes'] - 30),
                    player_data['AvgSessionDurationMinutes'] + 30
                )
                
                events.append({
                    'player_id': profile.player_id,
                    'event_type': 'session_start',
                    'timestamp': session_start,
                    'session_duration': duration
                })
                
                events.append({
                    'player_id': profile.player_id,
                    'event_type': 'session_end',
                    'timestamp': session_start + timedelta(minutes=duration),
                    'session_duration': duration
                })
                
                # Potential level completion during session
                if random.random() < 0.3:  # 30% chance of level completion per session
                    level = random.randint(1, player_data['PlayerLevel'])
                    events.append({
                        'player_id': profile.player_id,
                        'event_type': 'level_complete',
                        'timestamp': session_start + timedelta(minutes=random.randint(5, duration-5)),
                        'level': level
                    })
                
                # Potential purchase during session
                if profile.total_purchases > 0 and random.random() < 0.1:  # 10% chance
                    purchase_amounts = [0.99, 2.99, 4.99, 9.99, 19.99, 49.99]
                    amount = random.choice(purchase_amounts)
                    events.append({
                        'player_id': profile.player_id,
                        'event_type': 'purchase',
                        'timestamp': session_start + timedelta(minutes=random.randint(1, duration-1)),
                        'purchase_amount': amount
                    })
                
                session_count += 1
                current_date += timedelta(days=random.randint(1, 7))
            
            all_events.extend(events)
        
        return all_events
    
    def save_adapted_data(self, profiles: List[PlayerProfile], 
                         events: List[Dict[str, Any]], 
                         churn_features: List[ChurnFeatures],
                         output_dir: str = "data/real"):
        """Save the adapted data to files."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save profiles
        profiles_data = []
        for profile in profiles:
            profile_dict = asdict(profile)
            # Convert datetime objects to strings
            for key, value in profile_dict.items():
                if isinstance(value, datetime):
                    profile_dict[key] = value.isoformat()
            profiles_data.append(profile_dict)
        
        with open(f"{output_dir}/player_profiles.json", "w") as f:
            json.dump(profiles_data, f, indent=2)
        
        # Save events
        events_data = []
        for event in events:
            event_copy = event.copy()
            if isinstance(event_copy['timestamp'], datetime):
                event_copy['timestamp'] = event_copy['timestamp'].isoformat()
            events_data.append(event_copy)
        
        with open(f"{output_dir}/player_events.json", "w") as f:
            json.dump(events_data, f, indent=2)
        
        # Save churn features
        churn_data = []
        for features in churn_features:
            features_dict = asdict(features)
            # Convert date objects to strings
            for key, value in features_dict.items():
                if isinstance(value, date):
                    features_dict[key] = value.isoformat()
            churn_data.append(features_dict)
        
        with open(f"{output_dir}/churn_features.json", "w") as f:
            json.dump(churn_data, f, indent=2)
        
        print(f"Adapted data saved to {output_dir}/")
    
    def process_dataset(self, sample_size: int = None) -> Tuple[List[PlayerProfile], 
                                                              List[Dict[str, Any]], 
                                                              List[ChurnFeatures]]:
        """Process the entire dataset or a sample."""
        
        if sample_size and sample_size < len(self.df):
            print(f"Sampling {sample_size} players from {len(self.df)} total players...")
            self.df = self.df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        
        # Create all the data structures
        profiles = self.create_player_profiles()
        churn_features = self.create_churn_features(profiles)
        events = self.create_synthetic_events(profiles)
        
        print(f"Processing complete!")
        print(f"- {len(profiles)} player profiles")
        print(f"- {len(events)} events")
        print(f"- {len(churn_features)} churn feature sets")
        
        return profiles, events, churn_features


def main():
    """Process the real gaming dataset."""
    adapter = RealDataAdapter("online_gaming_behavior_dataset.csv")
    
    # Process a sample for testing
    print("Processing sample dataset (1000 players)...")
    profiles, events, churn_features = adapter.process_dataset(sample_size=1000)
    adapter.save_adapted_data(profiles, events, churn_features, "data/sample")
    
    # Process full dataset
    print("\nProcessing full dataset...")
    adapter = RealDataAdapter("online_gaming_behavior_dataset.csv")  # Reload full dataset
    profiles, events, churn_features = adapter.process_dataset()
    adapter.save_adapted_data(profiles, events, churn_features, "data/full")


if __name__ == "__main__":
    main()