-- Player Retention Analytics Database Schema

-- Player profiles table
CREATE TABLE player_profiles (
    player_id VARCHAR(50) PRIMARY KEY,
    registration_date TIMESTAMP NOT NULL,
    last_active_date TIMESTAMP,
    total_sessions INTEGER DEFAULT 0,
    total_playtime_minutes INTEGER DEFAULT 0,
    highest_level_reached INTEGER DEFAULT 1,
    total_purchases DECIMAL(10,2) DEFAULT 0.00,
    churn_risk_score DECIMAL(5,4),
    churn_prediction_date TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Player events table for raw event data
CREATE TABLE player_events (
    event_id SERIAL PRIMARY KEY,
    player_id VARCHAR(50) NOT NULL,
    event_type VARCHAR(50) NOT NULL,
    event_timestamp TIMESTAMP NOT NULL,
    event_data JSONB,
    session_id VARCHAR(50),
    level_reached INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (player_id) REFERENCES player_profiles(player_id)
);

-- Player sessions table
CREATE TABLE player_sessions (
    session_id VARCHAR(50) PRIMARY KEY,
    player_id VARCHAR(50) NOT NULL,
    start_time TIMESTAMP NOT NULL,
    end_time TIMESTAMP,
    duration_minutes INTEGER,
    level_reached INTEGER,
    events_count INTEGER DEFAULT 0,
    purchases_made DECIMAL(10,2) DEFAULT 0.00,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (player_id) REFERENCES player_profiles(player_id)
);

-- Retention metrics table for aggregated data
CREATE TABLE retention_metrics (
    id SERIAL PRIMARY KEY,
    cohort_date DATE NOT NULL,
    day_1_retention DECIMAL(5,4),
    day_7_retention DECIMAL(5,4),
    day_30_retention DECIMAL(5,4),
    cohort_size INTEGER NOT NULL,
    segment VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Churn features table for ML model features
CREATE TABLE churn_features (
    id SERIAL PRIMARY KEY,
    player_id VARCHAR(50) NOT NULL,
    feature_date DATE NOT NULL,
    days_since_last_session INTEGER,
    sessions_last_7_days INTEGER,
    avg_session_duration_minutes DECIMAL(8,2),
    levels_completed_last_week INTEGER,
    purchases_last_30_days DECIMAL(10,2),
    social_connections INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (player_id) REFERENCES player_profiles(player_id),
    UNIQUE(player_id, feature_date)
);

-- Create indexes for better query performance
CREATE INDEX idx_player_events_player_id ON player_events(player_id);
CREATE INDEX idx_player_events_timestamp ON player_events(event_timestamp);
CREATE INDEX idx_player_events_type ON player_events(event_type);
CREATE INDEX idx_player_sessions_player_id ON player_sessions(player_id);
CREATE INDEX idx_player_sessions_start_time ON player_sessions(start_time);
CREATE INDEX idx_retention_metrics_cohort_date ON retention_metrics(cohort_date);
CREATE INDEX idx_churn_features_player_id ON churn_features(player_id);
CREATE INDEX idx_churn_features_date ON churn_features(feature_date);

-- Create updated_at trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply trigger to player_profiles table
CREATE TRIGGER update_player_profiles_updated_at 
    BEFORE UPDATE ON player_profiles 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();