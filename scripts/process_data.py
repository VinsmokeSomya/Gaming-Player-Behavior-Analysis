"""
Script to process the real gaming dataset and convert it to our analytics format.
"""
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_adapter import RealDataAdapter


def main():
    """Process the real gaming dataset."""
    
    print("🎮 Real Gaming Dataset Processor")
    print("=" * 50)
    
    csv_path = "data/online_gaming_behavior_dataset.csv"
    
    if not os.path.exists(csv_path):
        print(f"❌ Dataset not found: {csv_path}")
        print("Please ensure the CSV file is in the project root directory.")
        return
    
    # Create adapter
    adapter = RealDataAdapter(csv_path, seed=42)
    
    # Process sample dataset first (for quick testing)
    print("\n📊 Processing sample dataset (1,000 players)...")
    profiles, events, churn_features = adapter.process_dataset(sample_size=1000)
    adapter.save_adapted_data(profiles, events, churn_features, "data/sample")
    
    # Show sample statistics
    print("\n📈 Sample Dataset Statistics:")
    print(f"- Players: {len(profiles)}")
    print(f"- Events: {len(events)}")
    print(f"- Avg Events per Player: {len(events) / len(profiles):.1f}")
    
    # Analyze engagement distribution
    engagement_dist = {}
    for i, profile in enumerate(profiles):
        original_data = adapter.df.iloc[i]
        engagement = original_data['EngagementLevel']
        engagement_dist[engagement] = engagement_dist.get(engagement, 0) + 1
    
    print(f"\n👥 Engagement Distribution:")
    for level, count in engagement_dist.items():
        print(f"- {level}: {count} ({count/len(profiles)*100:.1f}%)")
    
    # Revenue analysis
    total_revenue = sum(p.total_purchases for p in profiles)
    paying_players = sum(1 for p in profiles if p.total_purchases > 0)
    
    print(f"\n💰 Revenue Insights:")
    print(f"- Total Revenue: ${total_revenue:.2f}")
    print(f"- Paying Players: {paying_players} ({paying_players/len(profiles)*100:.1f}%)")
    print(f"- ARPU: ${total_revenue/len(profiles):.2f}")
    if paying_players > 0:
        print(f"- ARPPU: ${total_revenue/paying_players:.2f}")
    
    # Churn analysis
    high_risk = sum(1 for p in profiles if p.churn_risk_score > 0.7)
    medium_risk = sum(1 for p in profiles if 0.3 <= p.churn_risk_score <= 0.7)
    low_risk = sum(1 for p in profiles if p.churn_risk_score < 0.3)
    
    print(f"\n⚠️  Churn Risk Analysis:")
    print(f"- High Risk (>0.7): {high_risk} ({high_risk/len(profiles)*100:.1f}%)")
    print(f"- Medium Risk (0.3-0.7): {medium_risk} ({medium_risk/len(profiles)*100:.1f}%)")
    print(f"- Low Risk (<0.3): {low_risk} ({low_risk/len(profiles)*100:.1f}%)")
    
    # Ask if user wants to process full dataset
    print(f"\n🤔 The full dataset has {adapter.df.shape[0]} players.")
    response = input("Process the full dataset? This may take a few minutes. (y/N): ").strip().lower()
    
    if response in ['y', 'yes']:
        print("\n📊 Processing full dataset...")
        # Reload adapter for full dataset
        adapter = RealDataAdapter(csv_path, seed=42)
        profiles, events, churn_features = adapter.process_dataset()
        adapter.save_adapted_data(profiles, events, churn_features, "data/full")
        
        print(f"\n✅ Full dataset processing complete!")
        print(f"- {len(profiles)} player profiles")
        print(f"- {len(events)} events")
        print(f"- Average {len(events)/len(profiles):.1f} events per player")
    
    print("\n✅ Real data processing complete!")
    print("📁 Data saved to:")
    print("   - data/sample/ (1,000 players)")
    if response in ['y', 'yes']:
        print("   - data/full/ (40,034 players)")
    
    print("\n🚀 Ready to proceed with Task 3: ETL Pipeline Implementation!")


if __name__ == "__main__":
    main()