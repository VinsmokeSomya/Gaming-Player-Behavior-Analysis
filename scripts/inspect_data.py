"""
Script to inspect the processed real gaming dataset.
"""
import json
import sys
import os
from datetime import datetime
from collections import Counter

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))


def inspect_real_data(data_dir: str = "data/sample"):
    """Inspect the processed real gaming data."""
    
    print(f"🔍 REAL GAMING DATA INSPECTION: {data_dir}")
    print("=" * 60)
    
    # Load data
    with open(f"{data_dir}/player_profiles.json", "r") as f:
        profiles_data = json.load(f)
    
    with open(f"{data_dir}/player_events.json", "r") as f:
        events_data = json.load(f)
    
    with open(f"{data_dir}/churn_features.json", "r") as f:
        churn_data = json.load(f)
    
    print(f"📊 Dataset Overview:")
    print(f"- Players: {len(profiles_data)}")
    print(f"- Events: {len(events_data)}")
    print(f"- Churn Features: {len(churn_data)}")
    print(f"- Avg Events per Player: {len(events_data)/len(profiles_data):.1f}")
    
    # Analyze churn risk distribution
    risk_scores = [p['churn_risk_score'] for p in profiles_data]
    low_risk = sum(1 for r in risk_scores if r < 0.3)
    medium_risk = sum(1 for r in risk_scores if 0.3 <= r <= 0.7)
    high_risk = sum(1 for r in risk_scores if r > 0.7)
    
    print(f"\n⚠️  Churn Risk Distribution:")
    print(f"- Low Risk (<0.3): {low_risk} ({low_risk/len(profiles_data)*100:.1f}%)")
    print(f"- Medium Risk (0.3-0.7): {medium_risk} ({medium_risk/len(profiles_data)*100:.1f}%)")
    print(f"- High Risk (>0.7): {high_risk} ({high_risk/len(profiles_data)*100:.1f}%)")
    
    # Revenue analysis
    revenues = [p['total_purchases'] for p in profiles_data]
    paying_players = [r for r in revenues if r > 0]
    
    print(f"\n💰 Revenue Analysis:")
    print(f"- Total Revenue: ${sum(revenues):.2f}")
    print(f"- Paying Players: {len(paying_players)} ({len(paying_players)/len(profiles_data)*100:.1f}%)")
    if paying_players:
        print(f"- ARPU: ${sum(revenues)/len(profiles_data):.2f}")
        print(f"- ARPPU: ${sum(paying_players)/len(paying_players):.2f}")
        print(f"- Min Purchase: ${min(paying_players):.2f}")
        print(f"- Max Purchase: ${max(paying_players):.2f}")
    
    # Session analysis
    sessions = [p['total_sessions'] for p in profiles_data]
    playtime = [p['total_playtime_minutes'] for p in profiles_data]
    
    print(f"\n🎮 Engagement Analysis:")
    print(f"- Avg Sessions per Player: {sum(sessions)/len(sessions):.1f}")
    print(f"- Avg Playtime per Player: {sum(playtime)/len(playtime):.0f} minutes")
    print(f"- Avg Session Duration: {sum(playtime)/sum(sessions):.1f} minutes")
    
    # Level progression
    levels = [p['highest_level_reached'] for p in profiles_data]
    print(f"- Avg Level Reached: {sum(levels)/len(levels):.1f}")
    print(f"- Max Level Reached: {max(levels)}")
    
    # Event type distribution
    event_types = Counter(e['event_type'] for e in events_data)
    print(f"\n🎯 Event Distribution:")
    for event_type, count in event_types.most_common():
        print(f"- {event_type}: {count} ({count/len(events_data)*100:.1f}%)")
    
    # Recent activity analysis
    recent_sessions = [c['sessions_last_7_days'] for c in churn_data]
    active_players = sum(1 for s in recent_sessions if s > 0)
    
    print(f"\n📈 Recent Activity (Last 7 Days):")
    print(f"- Active Players: {active_players} ({active_players/len(churn_data)*100:.1f}%)")
    print(f"- Avg Sessions per Active Player: {sum(recent_sessions)/max(1, active_players):.1f}")
    
    # Inactivity analysis
    days_inactive = [c['days_since_last_session'] for c in churn_data]
    inactive_7_days = sum(1 for d in days_inactive if d > 7)
    inactive_30_days = sum(1 for d in days_inactive if d > 30)
    
    print(f"\n😴 Inactivity Analysis:")
    print(f"- Players Inactive >7 days: {inactive_7_days} ({inactive_7_days/len(churn_data)*100:.1f}%)")
    print(f"- Players Inactive >30 days: {inactive_30_days} ({inactive_30_days/len(churn_data)*100:.1f}%)")
    print(f"- Avg Days Since Last Session: {sum(days_inactive)/len(days_inactive):.1f}")
    
    print(f"\n✅ Real data inspection complete!")
    print(f"📁 This dataset is ready for retention analytics!")


def compare_datasets():
    """Compare synthetic vs real data characteristics."""
    print("\n🔄 SYNTHETIC vs REAL DATA COMPARISON")
    print("=" * 50)
    
    datasets = [
        ("Sample", "data/sample")
    ]
    
    for name, path in datasets:
        if os.path.exists(f"{path}/player_profiles.json"):
            with open(f"{path}/player_profiles.json", "r") as f:
                profiles = json.load(f)
            
            revenues = [p['total_purchases'] for p in profiles]
            paying_rate = sum(1 for r in revenues if r > 0) / len(revenues) * 100
            avg_sessions = sum(p['total_sessions'] for p in profiles) / len(profiles)
            
            print(f"\n{name}:")
            print(f"- Players: {len(profiles)}")
            print(f"- Paying Rate: {paying_rate:.1f}%")
            print(f"- Avg Sessions: {avg_sessions:.1f}")
            print(f"- Total Revenue: ${sum(revenues):.2f}")


def main():
    """Main inspection function."""
    
    # Inspect sample data
    if os.path.exists("data/sample"):
        inspect_real_data("data/sample")
    else:
        print("❌ Sample data not found. Run process_real_data.py first.")
        return
    
    # Compare with synthetic data
    compare_datasets()
    
    print("\n🚀 Ready to proceed with Task 3: ETL Pipeline!")


if __name__ == "__main__":
    main()