# Gaming Analytics Dataset

This directory contains real gaming data processed for the Player Retention Analytics system.

## Dataset Overview

The dataset is based on a real online gaming behavior dataset with 40,034 players, processed and adapted to work with our analytics models.

### Source Data
- **Original Dataset**: `online_gaming_behavior_dataset.csv` (40,034 players)
- **Source**: Real gaming behavior data with player demographics, engagement metrics, and behavioral patterns

### Processed Data Files

#### `/sample/` - Sample Dataset (1,000 players)
- `player_profiles.json` - Complete player profiles with metrics
- `player_events.json` - All player events (sessions, purchases, achievements)  
- `churn_features.json` - Churn prediction features for each player

#### `/full/` - Full Dataset (40,034 players)
- Same structure as sample dataset, full scale for production analytics

## Data Schema

### Player Profiles
```json
{
  "player_id": "player_abc123",
  "registration_date": "2024-07-15T10:30:00",
  "last_active_date": "2024-09-28T14:22:00",
  "total_sessions": 45,
  "total_playtime_minutes": 1350,
  "highest_level_reached": 28,
  "total_purchases": 89.97,
  "churn_risk_score": 0.35,
  "churn_prediction_date": "2024-09-30T12:00:00"
}
```

### Player Events
```json
{
  "player_id": "player_abc123",
  "event_type": "session_start|session_end|level_complete|purchase|achievement_unlock",
  "timestamp": "2024-09-28T14:22:00",
  "level": 15,  // for level_complete events
  "purchase_amount": 4.99,  // for purchase events
  "session_duration": 25  // for session events (minutes)
}
```

### Churn Features
```json
{
  "player_id": "player_abc123",
  "days_since_last_session": 2,
  "sessions_last_7_days": 3,
  "avg_session_duration_minutes": 22.5,
  "levels_completed_last_week": 2,
  "purchases_last_30_days": 9.99,
  "social_connections": 15,
  "feature_date": "2024-09-30"
}
```

## Key Metrics from Real Dataset (1,000 player sample)

### Player Engagement
- **Average Sessions per Player**: 169.9
- **Average Playtime**: ~17,250 minutes per player
- **Active Players (last 7 days)**: 45.5%

### Retention & Churn
- **Low Churn Risk**: 16.8% of players
- **Medium Churn Risk**: 70.0% of players  
- **High Churn Risk**: 13.2% of players
- **Players Inactive > 7 days**: 54.5%

### Revenue Metrics
- **Total Revenue**: $40,857.57
- **Paying Players**: 20.4%
- **ARPU** (Average Revenue Per User): $40.86
- **ARPPU** (Average Revenue Per Paying User): $200.28

### Event Distribution
- **Session Events**: 86.6% (start + end)
- **Level Completions**: 12.6% of all events
- **Purchases**: 0.8% of all events

### Engagement Distribution
- **High Engagement**: 25.7% of players
- **Medium Engagement**: 50.3% of players
- **Low Engagement**: 24.0% of players

## Usage

### Process Data
```bash
python scripts/process_data.py
```

### Inspect Data Quality
```bash
python scripts/inspect_data.py
```

### Load Data in Python
```python
import json
from datetime import datetime
from src.models import PlayerProfile, ChurnFeatures

# Load player profiles
with open("data/sample/player_profiles.json", "r") as f:
    profiles_data = json.load(f)

# Convert to model objects
profiles = []
for profile_data in profiles_data:
    # Convert ISO strings to datetime objects
    for field in ['registration_date', 'last_active_date', 'churn_prediction_date']:
        profile_data[field] = datetime.fromisoformat(profile_data[field])
    
    profiles.append(PlayerProfile.from_dict(profile_data))
```

## Data Quality Features

✅ **Real Gaming Behavior**: Authentic player engagement patterns from actual gaming data  
✅ **Temporal Consistency**: Events follow logical time sequences  
✅ **Business Logic Validation**: Churn risk correlates with actual activity patterns  
✅ **Comprehensive Coverage**: All required fields for retention analytics  
✅ **Large Scale**: 40,034 real players for robust analytics  
✅ **Data Integrity**: All relationships between players, events, and features are maintained  

This real gaming dataset provides an authentic foundation for developing and testing the player retention analytics system with actual player behavior patterns.