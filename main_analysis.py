#!/usr/bin/env python3
"""
Gaming Player Behavior Analysis - Main Analysis Pipeline

This script demonstrates:
• Player retention data analysis using SQL and statistical methods
• Churn prediction model achieving 80% accuracy using scikit-learn
• Visualizations showing player engagement patterns and drop-off points
• Technologies: Python, SQL, scikit-learn, Matplotlib
"""

import sys
import os
import sqlite3
from datetime import datetime

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_generator import create_database
from churn_prediction import ChurnPredictor
from visualizations import PlayerAnalyticsVisualizer

def run_sql_analysis(db_path):
    """Run SQL-based retention analysis"""
    print("=" * 60)
    print("RUNNING SQL-BASED RETENTION ANALYSIS")
    print("=" * 60)
    
    conn = sqlite3.connect(db_path)
    
    # Execute the SQL views
    with open('sql/retention_analysis.sql', 'r') as f:
        sql_commands = f.read().split(';')
        
    for command in sql_commands:
        command = command.strip()
        if command and not command.startswith('--'):
            try:
                conn.execute(command)
            except Exception as e:
                print(f"Warning: {e}")
    
    conn.commit()
    
    # Display key metrics from SQL analysis
    print("\n📊 KEY RETENTION METRICS:")
    
    # Daily active users trend
    dau_query = "SELECT AVG(daily_active_users) as avg_dau FROM daily_active_users"
    avg_dau = conn.execute(dau_query).fetchone()[0]
    print(f"• Average Daily Active Users: {avg_dau:.0f}")
    
    # Overall retention rates
    retention_query = """
    SELECT week_number, AVG(retention_rate) as avg_retention 
    FROM weekly_retention 
    GROUP BY week_number 
    ORDER BY week_number 
    LIMIT 4
    """
    retention_data = conn.execute(retention_query).fetchall()
    print("• Weekly Retention Rates:")
    for week, rate in retention_data:
        print(f"  Week {week}: {rate:.1f}%")
    
    # Churn statistics
    churn_query = "SELECT AVG(is_churned) * 100 as churn_rate FROM churned_players"
    churn_rate = conn.execute(churn_query).fetchone()[0]
    print(f"• Overall Churn Rate: {churn_rate:.1f}%")
    
    # Revenue metrics
    revenue_query = """
    SELECT 
        COUNT(*) as total_players,
        SUM(total_revenue) as total_revenue,
        AVG(total_revenue) as avg_revenue_per_player,
        COUNT(CASE WHEN total_revenue > 0 THEN 1 END) as paying_players
    FROM player_ltv
    """
    revenue_stats = conn.execute(revenue_query).fetchone()
    total_players, total_revenue, avg_revenue, paying_players = revenue_stats
    
    print(f"• Total Players: {total_players:,}")
    print(f"• Total Revenue: ${total_revenue:,.2f}")
    print(f"• Average Revenue per Player: ${avg_revenue:.2f}")
    print(f"• Paying Players: {paying_players:,} ({paying_players/total_players*100:.1f}%)")
    
    conn.close()
    
    return {
        'avg_dau': avg_dau,
        'churn_rate': churn_rate,
        'total_revenue': total_revenue,
        'paying_players_pct': paying_players/total_players*100
    }

def main():
    """Main analysis pipeline"""
    print("🎮 GAMING PLAYER BEHAVIOR ANALYSIS")
    print("=" * 60)
    print("Technologies: Python, SQL, scikit-learn, Matplotlib")
    print("=" * 60)
    
    # Define paths
    base_dir = '/home/runner/work/Gaming-Player-Behavior-Analysis/Gaming-Player-Behavior-Analysis'
    db_path = os.path.join(base_dir, 'data', 'gaming_data.db')
    models_dir = os.path.join(base_dir, 'models')
    viz_dir = os.path.join(base_dir, 'visualizations')
    
    # Step 1: Generate synthetic gaming data
    print("\n🏗️  STEP 1: GENERATING SYNTHETIC GAMING DATA")
    print("-" * 40)
    if not os.path.exists(db_path):
        create_database()
    else:
        print("Database already exists, using existing data...")
    
    # Step 2: SQL-based retention analysis
    print("\n📈 STEP 2: SQL-BASED RETENTION ANALYSIS")
    print("-" * 40)
    sql_metrics = run_sql_analysis(db_path)
    
    # Step 3: Machine Learning - Churn Prediction
    print("\n🤖 STEP 3: CHURN PREDICTION MODEL (TARGET: 80% ACCURACY)")
    print("-" * 50)
    predictor = ChurnPredictor(db_path)
    
    # Load and prepare data
    df = predictor.load_data()
    df = predictor.engineer_features(df)
    
    print(f"Dataset: {len(df):,} players")
    print(f"Churn rate: {df['is_churned'].mean():.1%}")
    
    # Train model
    X, y = predictor.prepare_features(df)
    accuracy, y_test, y_pred = predictor.train_model(X, y)
    
    # Save model
    predictor.save_model(models_dir)
    
    # Check if target achieved
    target_achieved = accuracy >= 0.80
    print(f"\n🎯 TARGET ACHIEVEMENT:")
    print(f"• Required Accuracy: 80.0%")
    print(f"• Achieved Accuracy: {accuracy:.1%}")
    print(f"• Target Met: {'✅ YES' if target_achieved else '❌ NO'}")
    
    # Step 4: Create visualizations
    print("\n📊 STEP 4: CREATING VISUALIZATIONS")
    print("-" * 40)
    print("Generating player engagement patterns and drop-off point visualizations...")
    
    visualizer = PlayerAnalyticsVisualizer(db_path, viz_dir)
    metrics, drop_offs = visualizer.create_all_visualizations()
    
    # Step 5: Summary Report
    print("\n📋 FINAL SUMMARY REPORT")
    print("=" * 60)
    
    print("✅ COMPLETED REQUIREMENTS:")
    print("• ✅ Analyzed player retention data using SQL and statistical methods")
    print(f"• {'✅' if target_achieved else '❌'} Built churn prediction model achieving 80% accuracy using scikit-learn")
    print("• ✅ Created visualizations showing player engagement patterns and drop-off points")
    print("• ✅ Technologies used: Python, SQL, scikit-learn, Matplotlib")
    
    print(f"\n📈 KEY FINDINGS:")
    print(f"• Average Daily Active Users: {sql_metrics['avg_dau']:.0f}")
    print(f"• Churn Rate: {sql_metrics['churn_rate']:.1f}%")
    print(f"• Total Revenue: ${sql_metrics['total_revenue']:,.2f}")
    print(f"• Paying Players: {sql_metrics['paying_players_pct']:.1f}%")
    print(f"• Model Accuracy: {accuracy:.1%}")
    print(f"• Major Drop-off Points: Levels {drop_offs}")
    
    print(f"\n📁 OUTPUT FILES:")
    print(f"• Database: {db_path}")
    print(f"• Models: {models_dir}/")
    print(f"• Visualizations: {viz_dir}/")
    
    print(f"\n🎯 PROJECT STATUS: {'COMPLETE ✅' if target_achieved else 'NEEDS IMPROVEMENT ⚠️'}")
    
    return {
        'sql_metrics': sql_metrics,
        'ml_accuracy': accuracy,
        'target_achieved': target_achieved,
        'drop_offs': drop_offs
    }

if __name__ == "__main__":
    results = main()