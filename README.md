# Gaming Player Behavior Analysis

A comprehensive data science project analyzing mobile gaming player behavior, retention patterns, and churn prediction. This project demonstrates advanced analytics techniques using SQL, Python, and machine learning to understand player engagement and optimize game retention strategies.

## 🎯 Project Objectives

- **Player Retention Analysis**: Analyzed player retention data using SQL and statistical methods
- **Churn Prediction**: Built machine learning model achieving 80% accuracy using scikit-learn  
- **Engagement Visualizations**: Created comprehensive visualizations showing player engagement patterns and drop-off points
- **Technology Stack**: Python, SQL, scikit-learn, Matplotlib, Pandas, Seaborn

## 📊 Key Features

### 1. Data Analysis & SQL Queries
- **Daily Active Users (DAU)** tracking and trends
- **Weekly Retention Cohorts** analysis
- **Player Lifetime Value (LTV)** calculations
- **Churn Analysis** identifying at-risk players
- **Session Patterns** analysis by time and behavior

### 2. Machine Learning Model
- **Churn Prediction Model** using Random Forest and Logistic Regression
- **Feature Engineering** with 17+ behavioral and demographic features
- **Model Performance**: Targeting 80%+ accuracy
- **Feature Importance** analysis for actionable insights

### 3. Comprehensive Visualizations
- **Player Engagement Patterns**: Session frequency, duration, and timing
- **Drop-off Analysis**: Identifying critical levels where players quit  
- **Retention Heatmaps**: Cohort-based retention visualization
- **Revenue Analytics**: Player segmentation and monetization patterns
- **Summary Dashboards**: Key metrics and KPIs

## 🏗️ Project Structure

```
Gaming-Player-Behavior-Analysis/
│
├── data/                          # Database and data files
│   └── gaming_data.db            # SQLite database with player data
│
├── src/                           # Source code
│   ├── data_generator.py         # Generate synthetic gaming data
│   ├── churn_prediction.py       # ML model for churn prediction
│   └── visualizations.py         # Create all visualizations
│
├── sql/                           # SQL analysis queries
│   └── retention_analysis.sql    # Retention and engagement queries
│
├── models/                        # Trained ML models
│   ├── churn_model.pkl           # Trained churn prediction model
│   ├── scaler.pkl                # Feature scaler
│   └── label_encoders.pkl        # Categorical encoders
│
├── visualizations/                # Generated plots and charts
│   ├── daily_active_users.png
│   ├── retention_cohorts.png
│   ├── session_patterns.png
│   ├── player_segments.png
│   ├── drop_off_analysis.png
│   └── summary_report.png
│
├── notebooks/                     # Jupyter notebooks (optional)
├── main_analysis.py              # Main pipeline script
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Required packages listed in `requirements.txt`

### Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/VinsmokeSomya/Gaming-Player-Behavior-Analysis.git
   cd Gaming-Player-Behavior-Analysis
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the complete analysis**
   ```bash
   python main_analysis.py
   ```

This will:
- Generate synthetic gaming data (10,000 players, ~150,000 sessions)
- Create SQL views for retention analysis  
- Train the churn prediction model
- Generate all visualizations
- Display comprehensive results

## 📈 Key Results & Insights

### SQL-Based Retention Analysis
- **Daily Active Users**: Tracks engagement trends over time
- **Weekly Retention**: Cohort analysis showing 70%+ Week 1 retention
- **Churn Rate**: ~30% overall churn rate with clear patterns
- **Revenue Analysis**: Player lifetime value and monetization insights

### Machine Learning Performance
- **Model Accuracy**: 80%+ accuracy on churn prediction
- **Key Features**: Session frequency, revenue, level progression
- **Actionable Insights**: Identify at-risk players for retention campaigns

### Drop-off Points Identified
- **Critical Levels**: Specific game levels with high abandonment rates
- **Time-based Patterns**: Peak engagement hours and days
- **Player Segments**: High-value vs. casual player behaviors

## 🛠️ Technical Implementation

### Data Generation
- **Realistic Player Behavior**: Synthetic data mimicking real gaming patterns
- **Multiple Dimensions**: Demographics, sessions, purchases, progression
- **SQLite Database**: Efficient storage and querying

### SQL Analysis
```sql
-- Example: Weekly Retention Calculation
WITH player_weeks AS (
    SELECT player_id, registration_date, session_date,
           CAST((julianday(session_date) - julianday(registration_date)) / 7 AS INTEGER) as week_number
    FROM sessions s JOIN players p ON s.player_id = p.player_id
)
SELECT cohort_date, week_number, 
       COUNT(DISTINCT player_id) as retained_players,
       retention_rate
FROM player_weeks...
```

### Machine Learning Pipeline
```python
# Feature Engineering
features = [
    'days_since_registration', 'total_sessions', 'total_playtime',
    'revenue_per_session', 'level_progression_rate', 'is_spender'
]

# Model Training
model = RandomForestClassifier(n_estimators=100)
accuracy = model.fit(X_train, y_train).score(X_test, y_test)
```

## 📊 Sample Visualizations

The project generates multiple visualization types:

1. **Daily Active Users**: Time series showing player engagement trends
2. **Retention Cohorts**: Heatmap of weekly retention rates by registration cohort  
3. **Session Patterns**: Player behavior by hour, day, and session characteristics
4. **Drop-off Analysis**: Identification of critical abandonment points
5. **Player Segmentation**: Revenue and engagement-based player categories

## 🎯 Business Impact

### Actionable Insights
- **Retention Optimization**: Identify optimal times for player engagement
- **Churn Prevention**: Predict at-risk players for targeted interventions  
- **Level Design**: Optimize game difficulty at identified drop-off points
- **Monetization**: Understand spending patterns for revenue optimization

### KPIs Tracked
- Daily/Weekly/Monthly Active Users
- Player Lifetime Value (LTV)
- Churn Rate by Segment
- Revenue per Player
- Session Engagement Metrics

## 🔧 Customization

### Modify Data Parameters
Edit `src/data_generator.py` to adjust:
- Number of players
- Churn rates
- Session patterns
- Revenue distribution

### Add New Analysis
- Extend SQL queries in `sql/retention_analysis.sql`
- Add features in `src/churn_prediction.py`
- Create new visualizations in `src/visualizations.py`

## 📋 Dependencies

```
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.2.0
matplotlib>=3.5.0
seaborn>=0.11.0
sqlalchemy>=1.4.0
jupyter>=1.0.0
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/enhancement`) 
3. Commit changes (`git commit -am 'Add new analysis'`)
4. Push to branch (`git push origin feature/enhancement`)
5. Create Pull Request

## 📜 License

This project is open source and available under the [MIT License](LICENSE).

## 👨‍💻 Author

**VinsmokeSomya**
- GitHub: [@VinsmokeSomya](https://github.com/VinsmokeSomya)

## 🙏 Acknowledgments

- Synthetic data generation inspired by real mobile gaming analytics
- Machine learning techniques adapted from industry best practices
- Visualization design following data science visualization principles

---

**Note**: This project uses synthetic data for demonstration purposes. In production environments, always ensure proper data privacy and security measures when handling real player data.