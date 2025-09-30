"""
Create visualizations showing player engagement patterns and drop-off points
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
from datetime import datetime
import os

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class PlayerAnalyticsVisualizer:
    def __init__(self, db_path, output_dir='visualizations'):
        self.db_path = db_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def load_data(self):
        """Load data from database"""
        conn = sqlite3.connect(self.db_path)
        
        # Load main datasets
        self.players = pd.read_sql_query("SELECT * FROM players", conn)
        self.sessions = pd.read_sql_query("SELECT * FROM sessions", conn)
        self.daily_active = pd.read_sql_query("SELECT * FROM daily_active_users", conn)
        self.retention = pd.read_sql_query("SELECT * FROM weekly_retention", conn)
        self.ltv = pd.read_sql_query("SELECT * FROM player_ltv", conn)
        self.churned = pd.read_sql_query("SELECT * FROM churned_players", conn)
        
        conn.close()
        
        # Convert date columns
        self.sessions['session_date'] = pd.to_datetime(self.sessions['session_date'])
        self.daily_active['date'] = pd.to_datetime(self.daily_active['date'])
        
    def plot_daily_active_users(self):
        """Plot Daily Active Users over time"""
        plt.figure(figsize=(12, 6))
        
        plt.plot(self.daily_active['date'], self.daily_active['daily_active_users'], 
                linewidth=2, marker='o', markersize=3)
        
        plt.title('Daily Active Users Over Time', fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Daily Active Users', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        # Add trend line
        x_numeric = np.arange(len(self.daily_active))
        z = np.polyfit(x_numeric, self.daily_active['daily_active_users'], 1)
        p = np.poly1d(z)
        plt.plot(self.daily_active['date'], p(x_numeric), "--", alpha=0.7, color='red')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/daily_active_users.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_retention_cohorts(self):
        """Plot retention cohorts heatmap"""
        # Prepare retention data for heatmap
        retention_pivot = self.retention.pivot_table(
            index='cohort_date', 
            columns='week_number', 
            values='retention_rate'
        )
        
        plt.figure(figsize=(14, 8))
        
        sns.heatmap(retention_pivot, 
                   annot=True, 
                   fmt='.1f', 
                   cmap='YlOrRd',
                   cbar_kws={'label': 'Retention Rate (%)'})
        
        plt.title('Weekly Retention Cohorts', fontsize=16, fontweight='bold')
        plt.xlabel('Week Number', fontsize=12)
        plt.ylabel('Registration Cohort', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/retention_cohorts.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_session_patterns(self):
        """Plot session patterns by time"""
        # Aggregate session data by hour
        self.sessions['hour'] = self.sessions['session_date'].dt.hour
        self.sessions['day_of_week'] = self.sessions['session_date'].dt.day_name()
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Sessions by hour of day
        hourly_sessions = self.sessions.groupby('hour').size()
        axes[0, 0].bar(hourly_sessions.index, hourly_sessions.values, color='skyblue')
        axes[0, 0].set_title('Sessions by Hour of Day')
        axes[0, 0].set_xlabel('Hour')
        axes[0, 0].set_ylabel('Number of Sessions')
        
        # Sessions by day of week
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily_sessions = self.sessions.groupby('day_of_week').size().reindex(day_order)
        axes[0, 1].bar(range(7), daily_sessions.values, color='lightcoral')
        axes[0, 1].set_title('Sessions by Day of Week')
        axes[0, 1].set_xlabel('Day')
        axes[0, 1].set_ylabel('Number of Sessions')
        axes[0, 1].set_xticks(range(7))
        axes[0, 1].set_xticklabels([day[:3] for day in day_order])
        
        # Session length distribution
        axes[1, 0].hist(self.sessions['session_length_minutes'], bins=30, color='lightgreen', alpha=0.7)
        axes[1, 0].set_title('Session Length Distribution')
        axes[1, 0].set_xlabel('Session Length (minutes)')
        axes[1, 0].set_ylabel('Frequency')
        
        # Level progression
        axes[1, 1].hist(self.sessions['level_reached'], bins=20, color='orange', alpha=0.7)
        axes[1, 1].set_title('Level Reached Distribution')
        axes[1, 1].set_xlabel('Level Reached')
        axes[1, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/session_patterns.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_player_segments(self):
        """Plot player segmentation based on LTV and engagement"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Revenue distribution
        revenue_data = self.ltv[self.ltv['total_revenue'] > 0]['total_revenue']
        axes[0, 0].hist(revenue_data, bins=30, color='gold', alpha=0.7)
        axes[0, 0].set_title('Player Revenue Distribution (Paying Players)')
        axes[0, 0].set_xlabel('Total Revenue ($)')
        axes[0, 0].set_ylabel('Number of Players')
        
        # Session count vs Revenue scatter
        axes[0, 1].scatter(self.ltv['total_sessions'], self.ltv['total_revenue'], 
                          alpha=0.6, color='purple')
        axes[0, 1].set_title('Sessions vs Revenue')
        axes[0, 1].set_xlabel('Total Sessions')
        axes[0, 1].set_ylabel('Total Revenue ($)')
        
        # Lifetime days distribution
        axes[1, 0].hist(self.ltv['lifetime_days'], bins=30, color='teal', alpha=0.7)
        axes[1, 0].set_title('Player Lifetime Distribution')
        axes[1, 0].set_xlabel('Lifetime (days)')
        axes[1, 0].set_ylabel('Number of Players')
        
        # Churn analysis
        churn_counts = self.churned['is_churned'].value_counts()
        axes[1, 1].pie(churn_counts.values, labels=['Active', 'Churned'], 
                      colors=['lightblue', 'salmon'], autopct='%1.1f%%')
        axes[1, 1].set_title('Player Churn Distribution')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/player_segments.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_drop_off_points(self):
        """Identify and visualize player drop-off points"""
        # Analyze level progression and drop-offs
        level_progression = self.sessions.groupby('player_id')['level_reached'].max().reset_index()
        level_counts = level_progression['level_reached'].value_counts().sort_index()
        
        # Calculate drop-off rates
        cumulative_players = level_counts.sort_index(ascending=False).cumsum().sort_index()
        total_players = len(level_progression)
        survival_rate = (cumulative_players / total_players * 100).iloc[:30]  # First 30 levels
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # Level completion rates
        axes[0].plot(survival_rate.index, survival_rate.values, marker='o', linewidth=2)
        axes[0].set_title('Player Survival Rate by Level (Drop-off Analysis)', fontsize=14)
        axes[0].set_xlabel('Level')
        axes[0].set_ylabel('% of Players Still Playing')
        axes[0].grid(True, alpha=0.3)
        
        # Highlight major drop-off points
        drop_offs = []
        for i in range(1, len(survival_rate)-1):
            if i in survival_rate.index:
                prev_rate = survival_rate.iloc[i-1] if i-1 in survival_rate.index else survival_rate.iloc[i]
                curr_rate = survival_rate.iloc[i]
                next_rate = survival_rate.iloc[i+1] if i+1 in survival_rate.index else curr_rate
                
                if prev_rate - curr_rate > 5:  # More than 5% drop
                    drop_offs.append(i)
                    axes[0].axvline(x=i, color='red', linestyle='--', alpha=0.7)
                    axes[0].annotate(f'Drop: Level {i}', 
                                   xy=(i, curr_rate), 
                                   xytext=(i+2, curr_rate+5),
                                   arrowprops=dict(arrowstyle='->', color='red', alpha=0.7))
        
        # Session frequency over time since registration
        player_sessions = self.sessions.merge(self.players, on='player_id')
        player_sessions['days_since_reg'] = (
            pd.to_datetime(player_sessions['session_date']) - 
            pd.to_datetime(player_sessions['registration_date'])
        ).dt.days
        
        # Bin by weeks
        player_sessions['weeks_since_reg'] = player_sessions['days_since_reg'] // 7
        weekly_sessions = player_sessions.groupby('weeks_since_reg').size()
        
        if len(weekly_sessions) > 0:
            axes[1].bar(weekly_sessions.index[:12], weekly_sessions.values[:12], color='lightblue')
            axes[1].set_title('Session Volume by Weeks Since Registration')
            axes[1].set_xlabel('Weeks Since Registration')
            axes[1].set_ylabel('Number of Sessions')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/drop_off_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return drop_offs
        
    def generate_summary_report(self):
        """Generate a summary statistics report"""
        # Calculate key metrics
        total_players = len(self.players)
        total_sessions = len(self.sessions)
        churn_rate = self.churned['is_churned'].mean()
        avg_session_length = self.sessions['session_length_minutes'].mean()
        total_revenue = self.sessions['revenue'].sum()
        paying_players = len(self.ltv[self.ltv['total_revenue'] > 0])
        
        # Create summary visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Key metrics as text boxes
        metrics = [
            ('Total Players', f'{total_players:,}'),
            ('Total Sessions', f'{total_sessions:,}'),
            ('Churn Rate', f'{churn_rate:.1%}'),
            ('Avg Session Length', f'{avg_session_length:.1f} min'),
            ('Total Revenue', f'${total_revenue:,.2f}'),
            ('Paying Players', f'{paying_players:,} ({paying_players/total_players:.1%})')
        ]
        
        for i, (metric, value) in enumerate(metrics):
            row, col = i // 3, i % 3
            axes[row, col].text(0.5, 0.5, f'{metric}\n{value}', 
                               horizontalalignment='center',
                               verticalalignment='center',
                               fontsize=16, fontweight='bold',
                               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
            axes[row, col].set_xlim(0, 1)
            axes[row, col].set_ylim(0, 1)
            axes[row, col].axis('off')
        
        plt.suptitle('Gaming Player Behavior Analysis - Key Metrics', fontsize=20, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/summary_report.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return metrics
        
    def create_all_visualizations(self):
        """Create all visualizations"""
        print("Loading data...")
        self.load_data()
        
        print("Creating Daily Active Users plot...")
        self.plot_daily_active_users()
        
        print("Creating Retention Cohorts heatmap...")
        self.plot_retention_cohorts()
        
        print("Creating Session Patterns plots...")
        self.plot_session_patterns()
        
        print("Creating Player Segments plots...")
        self.plot_player_segments()
        
        print("Creating Drop-off Analysis...")
        drop_offs = self.plot_drop_off_points()
        
        print("Generating Summary Report...")
        metrics = self.generate_summary_report()
        
        print(f"\nAll visualizations saved to: {self.output_dir}/")
        print(f"Key drop-off points identified at levels: {drop_offs}")
        
        return metrics, drop_offs

def main():
    """Main function to generate all visualizations"""
    db_path = '/home/runner/work/Gaming-Player-Behavior-Analysis/Gaming-Player-Behavior-Analysis/data/gaming_data.db'
    output_dir = '/home/runner/work/Gaming-Player-Behavior-Analysis/Gaming-Player-Behavior-Analysis/visualizations'
    
    visualizer = PlayerAnalyticsVisualizer(db_path, output_dir)
    metrics, drop_offs = visualizer.create_all_visualizations()
    
    return visualizer, metrics, drop_offs

if __name__ == "__main__":
    main()