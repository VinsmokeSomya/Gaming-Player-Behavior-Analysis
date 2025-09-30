"""
Generate synthetic gaming player behavior data for analysis
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sqlite3
import os

np.random.seed(42)

def generate_player_data(n_players=10000):
    """Generate synthetic player data"""
    
    # Player demographics
    player_ids = range(1, n_players + 1)
    countries = ['US', 'UK', 'CA', 'AU', 'DE', 'FR', 'JP', 'KR', 'BR', 'IN']
    platforms = ['iOS', 'Android', 'PC']
    
    # Generate registration dates (over 6 months)
    start_date = datetime.now() - timedelta(days=180)
    reg_dates = [start_date + timedelta(days=np.random.randint(0, 180)) for _ in range(n_players)]
    
    players_df = pd.DataFrame({
        'player_id': player_ids,
        'registration_date': reg_dates,
        'country': np.random.choice(countries, n_players),
        'platform': np.random.choice(platforms, n_players),
        'age_group': np.random.choice(['13-18', '19-25', '26-35', '36-45', '46+'], n_players)
    })
    
    return players_df

def generate_session_data(players_df, avg_sessions_per_player=15):
    """Generate session data for players"""
    sessions = []
    session_id = 1
    
    for _, player in players_df.iterrows():
        player_id = player['player_id']
        reg_date = player['registration_date']
        
        # Determine if player churns (30% churn rate)
        churns = np.random.random() < 0.3
        
        if churns:
            # Churned players have fewer sessions
            n_sessions = max(1, int(np.random.exponential(5)))
            active_days = min(30, int(np.random.exponential(10)))
        else:
            # Active players have more sessions
            n_sessions = max(5, int(np.random.exponential(avg_sessions_per_player)))
            active_days = min(60, int(np.random.exponential(30)))
        
        # Generate sessions over active period
        for _ in range(n_sessions):
            session_date = reg_date + timedelta(days=np.random.randint(0, active_days))
            session_length = max(1, int(np.random.exponential(20)))  # minutes
            level_reached = max(1, int(np.random.exponential(10)))
            purchases = np.random.poisson(0.1)  # Low purchase rate
            
            sessions.append({
                'session_id': session_id,
                'player_id': player_id,
                'session_date': session_date,
                'session_length_minutes': session_length,
                'level_reached': level_reached,
                'purchases_made': purchases,
                'revenue': purchases * np.random.exponential(5.0) if purchases > 0 else 0
            })
            session_id += 1
    
    return pd.DataFrame(sessions)

def create_database():
    """Create SQLite database with player and session data"""
    db_path = '/home/runner/work/Gaming-Player-Behavior-Analysis/Gaming-Player-Behavior-Analysis/data/gaming_data.db'
    
    # Generate data
    print("Generating player data...")
    players_df = generate_player_data()
    
    print("Generating session data...")
    sessions_df = generate_session_data(players_df)
    
    # Create database
    print("Creating database...")
    conn = sqlite3.connect(db_path)
    
    # Store data
    players_df.to_sql('players', conn, if_exists='replace', index=False)
    sessions_df.to_sql('sessions', conn, if_exists='replace', index=False)
    
    # Create indexes for better query performance
    conn.execute('CREATE INDEX idx_player_id ON sessions(player_id)')
    conn.execute('CREATE INDEX idx_session_date ON sessions(session_date)')
    
    conn.close()
    
    print(f"Database created at: {db_path}")
    print(f"Players: {len(players_df)}")
    print(f"Sessions: {len(sessions_df)}")
    
    return db_path

if __name__ == "__main__":
    create_database()