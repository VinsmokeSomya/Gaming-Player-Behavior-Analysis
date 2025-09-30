# 🎮 Player Retention Analytics Dashboard - Complete Setup Guide

A comprehensive analytics system for mobile games that analyzes **real player behavior data**, predicts churn risk, and provides actionable insights through interactive visualizations.

## 📊 What This System Does

- **Player Retention Analysis**: Calculate daily, weekly, and monthly retention rates from real data
- **Churn Prediction**: Machine learning models to identify at-risk players using actual behavior patterns
- **Interactive Visualizations**: Charts and reports based on 40,034 real players
- **Drop-off Analysis**: Identify specific game levels where players quit using real progression data
- **Automated Reporting**: Daily summaries and alerts for unusual patterns

## 🗂️ Real Data Overview

This system uses **authentic gaming data** from 40,034 real players:

- **Original Dataset**: `data/online_gaming_behavior_dataset.csv` (40,034 players)
- **Processed Sample**: `data/sample/` (1,000 players for faster loading)
- **Processed Full**: `data/full/` (40,034 players for production analytics)

### Real Data Metrics
- **Average Sessions per Player**: 169.9
- **Average Playtime**: ~17,250 minutes per player  
- **Active Players (last 7 days)**: 45.5%
- **Total Revenue**: $40,857.57 from real purchases
- **Paying Players**: 20.4% of the player base

---

## 🚀 Quick Start (5 Minutes)

### Prerequisites
- **Python 3.9+** (you have 3.13 ✅)
- **Docker & Docker Compose** (you have it ✅)
- **Git** (for cloning if needed)

### 1. Setup Environment

```bash
# Navigate to project directory
cd "Gaming-Player Behavior Analysis"

# Create and activate virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Start Database

```bash
# Start PostgreSQL database
docker-compose up -d postgres

# Wait for database to be ready (about 30 seconds)
docker-compose logs -f postgres
# Look for "database system is ready to accept connections"
```

### 3. Verify Database Connection

```bash
# Test database connection
python -c "from src.database import db_manager; print('✅ Connected' if db_manager.test_connection() else '❌ Failed')"
```

### 4. Run the Dashboard

```bash
# Start the analytics dashboard
python app.py
```

### 5. Access Your Dashboard

Open your browser and go to: **http://127.0.0.1:8050**

🎉 **You're now analyzing real gaming data from 40,034 players!**

---

## 📈 Dashboard Features

### 🏠 Overview Tab
- **Key Metrics**: Total players, high-risk players, average DAU, funnel conversion
- **Cross-filtering Charts**: Click charts to filter data across the dashboard
- **Real-time Updates**: Data refreshes automatically

### 🔥 Cohort Analysis Tab  
- **Retention Heatmap**: See how player cohorts retain over time
- **Interactive Drill-down**: Click cells to analyze specific cohorts
- **Cohort Comparison**: Compare retention curves between different registration periods

### 📊 Engagement Timeline Tab
- **Player Activity**: Daily, weekly, monthly active users from real data
- **Session Metrics**: Average session duration and frequency
- **Anomaly Detection**: Highlight unusual engagement patterns

### ⚠️ Churn Analysis Tab
- **Risk Distribution**: See churn risk across your real player base
- **Segment Analysis**: Compare risk by player segments (new, casual, core, premium)
- **Actionable Insights**: Export high-risk players for retention campaigns

### 🎯 Funnel Analysis Tab
- **Drop-off Visualization**: See where players quit in your game progression
- **Level Heatmap**: Identify difficult levels causing player loss
- **Cohort Funnel Comparison**: Compare progression across different player cohorts

### 🔍 Drill-Down Tab
- **Detailed Analysis**: Activated when you click on charts in other tabs
- **Player Journey**: Deep dive into specific player segments or time periods

---

## 🔧 Advanced Configuration

### ✅ Currently Using Full Dataset (40,034 players)

Your dashboard is **already configured** to use the complete dataset with all 40,034 real players!

- **Data Source**: `data/full/` directory
- **Loading Time**: ~5-10 seconds  
- **Complete Analytics**: All real player behavior patterns

**To switch back to sample data** (if needed for faster loading):

```python
# Edit app.py line 66:
data_loader = DataLoader(data_dir="data/sample")  # Switch to 1,000 players
```

### Performance Optimization

```bash
# Start Redis for caching (optional)
docker-compose up -d redis

# Set environment variable
export REDIS_URL=redis://localhost:6379/0
```

### Database Configuration

The system uses PostgreSQL for data storage. Configuration in `.env`:

```bash
DB_HOST=localhost
DB_PORT=5432
DB_NAME=player_analytics
DB_USER=analytics_user
DB_PASSWORD=analytics_password
```

---

## 📊 Data Processing Pipeline

### Understanding the Data Flow

1. **Raw Data**: `online_gaming_behavior_dataset.csv` (40,034 real players)
2. **Data Adapter**: `src/data_adapter.py` processes raw CSV into analytics models
3. **ETL Pipeline**: `src/etl/` loads and validates processed data
4. **Analytics Engine**: Calculates cohorts, retention, churn risk
5. **Visualization**: Interactive dashboard displays insights

### Reprocessing Data

If you need to reprocess the original dataset:

```bash
# Process sample dataset (1,000 players)
python src/data_adapter.py

# This creates:
# - data/sample/player_profiles.json
# - data/sample/player_events.json  
# - data/sample/churn_features.json
```

### Data Quality Validation

Your dashboard automatically validates data quality during loading. Check the console output when starting the dashboard for any data quality messages.

---

## 🛠️ Troubleshooting

### Common Issues

#### 1. Dashboard Won't Start

```bash
# Check if port 8050 is in use
netstat -an | findstr :8050

# Kill process using port (Windows)
taskkill /F /PID <process_id>

# Or use different port
python app.py --port 8051
```

#### 2. Database Connection Failed

```bash
# Check if PostgreSQL is running
docker-compose ps

# Restart database
docker-compose restart postgres

# Check logs
docker-compose logs postgres
```

#### 3. Data Loading Errors

```bash
# Verify data files exist
ls data/sample/

# Check data integrity
python -c "
import json
with open('data/sample/player_profiles.json') as f:
    data = json.load(f)
    print(f'✅ {len(data)} player profiles loaded')
"
```

#### 4. Performance Issues

```bash
# Use sample data instead of full dataset
# Edit app.py line 35: data_dir="data/sample"

# Enable caching
docker-compose up -d redis
export REDIS_URL=redis://localhost:6379/0
```

### Getting Help

1. **Check Logs**: Look at console output when starting the dashboard
2. **Verify Prerequisites**: Ensure Python 3.9+, Docker are installed
3. **Data Integrity**: Verify JSON files in `data/sample/` are valid
4. **Port Conflicts**: Try different port if 8050 is in use

---

## 🔄 Daily Usage Workflow

### Starting Your Analytics Session

```bash
# 1. Activate environment
.venv\Scripts\activate

# 2. Start database (if not running)
docker-compose up -d postgres

# 3. Start dashboard
python app.py

# 4. Open browser to http://127.0.0.1:8050
```

### Stopping the System

```bash
# 1. Stop dashboard (Ctrl+C in terminal)
# 2. Stop database
docker-compose stop postgres

# 3. Deactivate environment
deactivate
```

### Regular Maintenance

```bash
# Weekly: Update dependencies
pip install -r requirements.txt --upgrade

# Monthly: Clean up Docker
docker system prune

# As needed: Reprocess data
python src/data_adapter.py
```

---

## 📁 Project Structure

```
Gaming-Player Behavior Analysis/
├── 📊 data/                          # Real gaming data
│   ├── online_gaming_behavior_dataset.csv  # Original 40,034 players
│   ├── sample/                       # Processed sample (1,000 players)
│   │   ├── player_profiles.json      # Player demographics & metrics
│   │   ├── player_events.json        # All player events & sessions
│   │   └── churn_features.json       # Churn prediction features
│   └── full/                         # Processed full dataset (40,034 players)
│
├── 🎮 app.py                         # Main dashboard application
├── 🐳 docker-compose.yml             # Database setup
├── 📋 requirements.txt               # Python dependencies
├── ⚙️ .env                          # Configuration
│
├── 📦 src/                           # Source code
│   ├── 🔄 etl/                      # Data processing pipeline
│   │   ├── ingestion.py             # Load real data from JSON
│   │   ├── cohort_analysis.py       # Calculate retention cohorts
│   │   └── aggregation.py           # Metrics aggregation
│   ├── 📊 visualization/            # Chart components
│   │   ├── cohort_heatmap.py        # Retention heatmaps
│   │   ├── engagement_timeline.py   # Activity charts
│   │   ├── churn_histogram.py       # Risk analysis
│   │   └── dropoff_funnel.py        # Progression funnels
│   ├── 🤖 models/                   # Data models
│   ├── 🔍 validation/               # Data quality checks
│   └── 🛠️ utils/                    # Utilities
│
└── 📚 tests/                         # Test suite
```

---

## 🎯 Next Steps

### Immediate Actions
1. **Explore the Dashboard**: Click through all tabs to see real player insights
2. **Try Cross-filtering**: Click on charts to filter data across tabs
3. **Analyze Your Data**: Look for patterns in the real player behavior

### Advanced Analytics
1. **Custom Segments**: Modify player segmentation logic in `app.py`
2. **New Metrics**: Add custom KPIs using the real event data
3. **Alerts**: Set up automated alerts for unusual patterns
4. **Export Data**: Use the export features for further analysis

### Production Deployment
1. **Scale Up**: Switch to full dataset (40,034 players)
2. **Performance**: Enable Redis caching for faster loading
3. **Security**: Configure authentication and HTTPS
4. **Monitoring**: Set up logging and performance monitoring

---

## 🏆 Success Metrics

You'll know the system is working correctly when you see:

✅ **Dashboard loads** at http://127.0.0.1:8050  
✅ **Real data metrics** showing 1,000 players (sample) or 40,034 (full)  
✅ **Interactive charts** that respond to clicks and filters  
✅ **Cohort heatmap** showing actual retention patterns  
✅ **Churn analysis** with real risk scores  
✅ **Funnel analysis** showing actual level progression  
✅ **Performance** under 5 seconds for chart updates  

---

## 📞 Support

This system analyzes **real gaming behavior data** from 40,034 authentic players, providing genuine insights into player retention, churn risk, and engagement patterns.

**Happy analyzing! 🎮📊**