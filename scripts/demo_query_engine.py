"""Demonstration script for the retention query engine.

This script shows how to use the various query functions in the retention query engine.
Run this script after setting up the database and loading some sample data.
"""

import sys
import os
from datetime import date, datetime, timedelta

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.analytics.query_engine import query_engine
from src.database import db_manager


def main():
    """Demonstrate query engine functionality."""
    print("=== Player Retention Analytics Query Engine Demo ===\n")
    
    # Test database connection
    print("1. Testing database connection...")
    try:
        if db_manager.test_connection():
            print("✓ Database connection successful")
        else:
            print("✗ Database connection failed")
            print("Please ensure PostgreSQL is running with docker-compose up")
            return
    except Exception as e:
        print(f"✗ Database connection error: {e}")
        print("Please ensure PostgreSQL is running with docker-compose up")
        return
    
    # Define date range for analysis
    end_date = date.today()
    start_date = end_date - timedelta(days=30)
    
    print(f"\nAnalyzing data from {start_date} to {end_date}\n")
    
    # Demo 1: Calculate cohort retention
    print("2. Calculating cohort retention rates...")
    try:
        retention_results = query_engine.calculate_cohort_retention(start_date, end_date)
        
        if retention_results:
            print(f"Found {len(retention_results)} cohorts:")
            for result in retention_results[:5]:  # Show first 5 results
                print(f"  Cohort {result.cohort_date}: "
                      f"Day 1: {result.day_1_retention:.2%}, "
                      f"Day 7: {result.day_7_retention:.2%}, "
                      f"Day 30: {result.day_30_retention:.2%} "
                      f"(Size: {result.cohort_size})")
        else:
            print("  No cohort data found (database may be empty)")
    except Exception as e:
        print(f"  Error: {e}")
    
    # Demo 2: Analyze drop-off by level
    print("\n3. Analyzing drop-off rates by game level...")
    try:
        dropoff_results = query_engine.analyze_drop_off_by_level(start_date, end_date, max_level=10)
        
        if dropoff_results:
            print(f"Found drop-off data for {len(dropoff_results)} levels:")
            for result in dropoff_results[:5]:  # Show first 5 levels
                print(f"  Level {result.level}: "
                      f"{result.players_reached} reached, "
                      f"{result.players_completed} completed "
                      f"(Drop-off: {result.drop_off_rate:.2%})")
        else:
            print("  No level progression data found")
    except Exception as e:
        print(f"  Error: {e}")
    
    # Demo 3: Segment players by behavior
    print("\n4. Segmenting players by behavior...")
    try:
        segment_results = query_engine.segment_players_by_behavior(start_date, end_date)
        
        if segment_results:
            print("Player segments:")
            for result in segment_results:
                print(f"  {result.segment}: {result.player_count} players, "
                      f"Avg sessions: {result.avg_sessions:.1f}, "
                      f"Avg playtime: {result.avg_playtime:.0f}min, "
                      f"Day 7 retention: {result.avg_retention_day_7:.2%}")
        else:
            print("  No player data found for segmentation")
    except Exception as e:
        print(f"  Error: {e}")
    
    # Demo 4: Get retention for specific segment
    print("\n5. Getting retention for High Engagement segment...")
    try:
        high_engagement_retention = query_engine.get_retention_by_segment(
            start_date, end_date, 'High Engagement'
        )
        
        if high_engagement_retention:
            print(f"High Engagement retention for {len(high_engagement_retention)} cohorts:")
            for result in high_engagement_retention[:3]:  # Show first 3
                print(f"  Cohort {result.cohort_date}: "
                      f"Day 1: {result.day_1_retention:.2%}, "
                      f"Day 7: {result.day_7_retention:.2%}, "
                      f"Day 30: {result.day_30_retention:.2%}")
        else:
            print("  No high engagement cohort data found")
    except Exception as e:
        print(f"  Error: {e}")
    
    # Demo 5: Get daily active users
    print("\n6. Getting daily active users...")
    try:
        # Get last 7 days of DAU data
        dau_start = end_date - timedelta(days=7)
        dau_results = query_engine.get_daily_active_users(dau_start, end_date)
        
        if dau_results:
            print(f"Daily active users for last {len(dau_results)} days:")
            for activity_date, dau_count in dau_results:
                print(f"  {activity_date}: {dau_count} active users")
        else:
            print("  No daily activity data found")
    except Exception as e:
        print(f"  Error: {e}")
    
    # Demo 6: Get weekly active users
    print("\n7. Getting weekly active users...")
    try:
        # Get last 4 weeks of WAU data
        wau_start = end_date - timedelta(weeks=4)
        wau_results = query_engine.get_weekly_active_users(wau_start, end_date)
        
        if wau_results:
            print(f"Weekly active users for last {len(wau_results)} weeks:")
            for week_start, wau_count in wau_results:
                print(f"  Week of {week_start}: {wau_count} active users")
        else:
            print("  No weekly activity data found")
    except Exception as e:
        print(f"  Error: {e}")
    
    print("\n=== Demo completed ===")
    print("\nNote: If no data was found, you may need to:")
    print("1. Load sample data into the database")
    print("2. Run the ETL pipeline to process player events")
    print("3. Ensure the date range includes periods with player activity")


if __name__ == "__main__":
    main()