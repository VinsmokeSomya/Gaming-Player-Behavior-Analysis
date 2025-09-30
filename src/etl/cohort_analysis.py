"""
Cohort analysis data transformation pipeline for player retention analytics.
"""
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

from ..models import PlayerProfile, RetentionMetrics


class CohortAnalyzer:
    """Handles cohort analysis transformations for retention analytics."""
    
    def __init__(self):
        """Initialize cohort analyzer."""
        pass
    
    def create_cohort_table(self, events_df: pd.DataFrame, 
                          profiles: List[PlayerProfile]) -> pd.DataFrame:
        """Create a cohort table showing retention rates over time."""
        
        # Prepare profiles data
        profiles_data = []
        for profile in profiles:
            profiles_data.append({
                'player_id': profile.player_id,
                'registration_date': profile.registration_date.date()
            })
        
        profiles_df = pd.DataFrame(profiles_data)
        
        # Prepare events data
        events_df = events_df.copy()
        events_df['date'] = pd.to_datetime(events_df['timestamp']).dt.date
        
        # Get unique player activity dates
        player_activity = events_df.groupby(['player_id', 'date']).size().reset_index(name='event_count')
        player_activity = player_activity[['player_id', 'date']].drop_duplicates()
        
        # Merge with registration dates
        cohort_data = player_activity.merge(profiles_df, on='player_id', how='left')
        cohort_data = cohort_data.dropna()
        
        # Calculate days since registration
        cohort_data['days_since_registration'] = (
            pd.to_datetime(cohort_data['date']) - pd.to_datetime(cohort_data['registration_date'])
        ).dt.days
        
        # Create cohort table
        cohort_table = cohort_data.pivot_table(
            index='registration_date',
            columns='days_since_registration',
            values='player_id',
            aggfunc='nunique',
            fill_value=0
        )
        
        # Calculate cohort sizes (day 0 registrations)
        cohort_sizes = profiles_df.groupby('registration_date').size()
        
        # Convert to retention rates
        cohort_retention = cohort_table.divide(cohort_sizes, axis=0)
        
        return cohort_retention
    
    def create_cohort_heatmap_data(self, events_df: pd.DataFrame,
                                 profiles: List[PlayerProfile]) -> Dict[str, Any]:
        """Create data structure optimized for heatmap visualization."""
        
        cohort_table = self.create_cohort_table(events_df, profiles)
        
        # Prepare data for heatmap
        heatmap_data = {
            'cohort_dates': cohort_table.index.tolist(),
            'day_columns': cohort_table.columns.tolist(),
            'retention_matrix': cohort_table.values.tolist(),
            'cohort_sizes': []
        }
        
        # Add cohort sizes
        for cohort_date in cohort_table.index:
            cohort_size = cohort_table.loc[cohort_date, 0] if 0 in cohort_table.columns else 0
            heatmap_data['cohort_sizes'].append(cohort_size)
        
        return heatmap_data
    
    def analyze_cohort_trends(self, events_df: pd.DataFrame,
                            profiles: List[PlayerProfile]) -> Dict[str, Any]:
        """Analyze trends across cohorts to identify patterns."""
        
        cohort_table = self.create_cohort_table(events_df, profiles)
        
        trends = {
            'average_retention_by_day': {},
            'cohort_performance': [],
            'retention_decline_rates': {},
            'best_performing_cohorts': [],
            'worst_performing_cohorts': []
        }
        
        # Calculate average retention by day across all cohorts
        for day in cohort_table.columns:
            if day >= 0:  # Only include valid days
                avg_retention = cohort_table[day].mean()
                trends['average_retention_by_day'][int(day)] = float(avg_retention)
        
        # Analyze individual cohort performance
        for cohort_date in cohort_table.index:
            cohort_row = cohort_table.loc[cohort_date]
            
            # Calculate key retention metrics for this cohort
            day_1_retention = cohort_row.get(1, 0)
            day_7_retention = cohort_row.get(7, 0)
            day_30_retention = cohort_row.get(30, 0)
            
            cohort_performance = {
                'cohort_date': cohort_date.isoformat(),
                'day_1_retention': float(day_1_retention),
                'day_7_retention': float(day_7_retention),
                'day_30_retention': float(day_30_retention),
                'cohort_size': int(cohort_row.get(0, 0))
            }
            
            trends['cohort_performance'].append(cohort_performance)
        
        # Calculate retention decline rates
        key_days = [1, 7, 30]
        for i in range(len(key_days) - 1):
            from_day = key_days[i]
            to_day = key_days[i + 1]
            
            if from_day in cohort_table.columns and to_day in cohort_table.columns:
                decline_rates = []
                for cohort_date in cohort_table.index:
                    from_retention = cohort_table.loc[cohort_date, from_day]
                    to_retention = cohort_table.loc[cohort_date, to_day]
                    
                    if from_retention > 0:
                        decline_rate = (from_retention - to_retention) / from_retention
                        decline_rates.append(decline_rate)
                
                if decline_rates:
                    avg_decline = np.mean(decline_rates)
                    trends['retention_decline_rates'][f'day_{from_day}_to_{to_day}'] = float(avg_decline)
        
        # Identify best and worst performing cohorts
        if 30 in cohort_table.columns:
            cohort_30_day = cohort_table[30].dropna()
            
            # Best performing cohorts (top 20%)
            top_threshold = cohort_30_day.quantile(0.8)
            best_cohorts = cohort_30_day[cohort_30_day >= top_threshold]
            
            for cohort_date, retention in best_cohorts.items():
                trends['best_performing_cohorts'].append({
                    'cohort_date': cohort_date.isoformat(),
                    'day_30_retention': float(retention)
                })
            
            # Worst performing cohorts (bottom 20%)
            bottom_threshold = cohort_30_day.quantile(0.2)
            worst_cohorts = cohort_30_day[cohort_30_day <= bottom_threshold]
            
            for cohort_date, retention in worst_cohorts.items():
                trends['worst_performing_cohorts'].append({
                    'cohort_date': cohort_date.isoformat(),
                    'day_30_retention': float(retention)
                })
        
        return trends
    
    def create_lifecycle_segments(self, events_df: pd.DataFrame,
                                profiles: List[PlayerProfile]) -> pd.DataFrame:
        """Create player lifecycle segments based on activity patterns."""
        
        # Prepare data
        events_df = events_df.copy()
        events_df['date'] = pd.to_datetime(events_df['timestamp']).dt.date
        
        # Calculate player activity metrics
        player_metrics = []
        
        for profile in profiles:
            player_events = events_df[events_df['player_id'] == profile.player_id]
            
            if len(player_events) == 0:
                # Inactive player
                segment = 'inactive'
                days_since_last_activity = (date.today() - profile.registration_date.date()).days
                total_active_days = 0
                avg_events_per_day = 0
            else:
                # Calculate activity metrics
                player_dates = player_events['date'].unique()
                total_active_days = len(player_dates)
                last_activity_date = max(player_dates)
                days_since_last_activity = (date.today() - last_activity_date).days
                
                total_events = len(player_events)
                registration_days = (date.today() - profile.registration_date.date()).days + 1
                avg_events_per_day = total_events / registration_days if registration_days > 0 else 0
                
                # Determine lifecycle segment
                if days_since_last_activity <= 1:
                    if avg_events_per_day >= 5:
                        segment = 'highly_active'
                    elif avg_events_per_day >= 1:
                        segment = 'active'
                    else:
                        segment = 'casual'
                elif days_since_last_activity <= 7:
                    segment = 'at_risk'
                elif days_since_last_activity <= 30:
                    segment = 'dormant'
                else:
                    segment = 'churned'
            
            player_metrics.append({
                'player_id': profile.player_id,
                'registration_date': profile.registration_date.date(),
                'last_activity_date': last_activity_date if 'last_activity_date' in locals() else None,
                'days_since_last_activity': days_since_last_activity,
                'total_active_days': total_active_days,
                'avg_events_per_day': avg_events_per_day,
                'lifecycle_segment': segment
            })
        
        return pd.DataFrame(player_metrics)
    
    def calculate_cohort_ltv(self, events_df: pd.DataFrame,
                           profiles: List[PlayerProfile]) -> pd.DataFrame:
        """Calculate lifetime value metrics by cohort."""
        
        # Extract purchase events
        purchase_events = events_df[events_df['event_type'] == 'purchase'].copy()
        
        if purchase_events.empty:
            # Return empty DataFrame if no purchase data
            return pd.DataFrame(columns=[
                'cohort_date', 'cohort_size', 'total_revenue', 
                'avg_ltv', 'paying_users', 'conversion_rate'
            ])
        
        # Prepare profiles data
        profiles_data = []
        for profile in profiles:
            profiles_data.append({
                'player_id': profile.player_id,
                'registration_date': profile.registration_date.date()
            })
        
        profiles_df = pd.DataFrame(profiles_data)
        
        # Calculate revenue by player and cohort
        purchase_events['revenue'] = purchase_events['purchase_amount']
        player_revenue = purchase_events.groupby('player_id')['revenue'].sum().reset_index()
        
        # Merge with cohort data
        cohort_revenue = player_revenue.merge(profiles_df, on='player_id', how='right')
        cohort_revenue['revenue'] = cohort_revenue['revenue'].fillna(0)
        
        # Calculate cohort LTV metrics
        cohort_ltv = cohort_revenue.groupby('registration_date').agg({
            'player_id': 'count',  # cohort_size
            'revenue': ['sum', 'mean', lambda x: (x > 0).sum()]  # total, avg, paying users
        }).reset_index()
        
        # Flatten column names
        cohort_ltv.columns = [
            'cohort_date', 'cohort_size', 'total_revenue', 'avg_ltv', 'paying_users'
        ]
        
        # Calculate conversion rate
        cohort_ltv['conversion_rate'] = cohort_ltv['paying_users'] / cohort_ltv['cohort_size']
        
        return cohort_ltv
    
    def create_retention_funnel(self, events_df: pd.DataFrame,
                              profiles: List[PlayerProfile],
                              funnel_days: List[int] = [1, 3, 7, 14, 30]) -> Dict[str, Any]:
        """Create retention funnel showing drop-off at each stage."""
        
        cohort_table = self.create_cohort_table(events_df, profiles)
        
        funnel_data = {
            'stages': [],
            'total_players': len(profiles),
            'retention_rates': [],
            'drop_off_rates': []
        }
        
        # Calculate retention at each funnel stage
        prev_retention = 1.0  # Start with 100% at registration
        
        for day in funnel_days:
            if day in cohort_table.columns:
                # Calculate average retention across all cohorts
                avg_retention = cohort_table[day].mean()
                drop_off = prev_retention - avg_retention
                
                funnel_data['stages'].append(f'Day {day}')
                funnel_data['retention_rates'].append(float(avg_retention))
                funnel_data['drop_off_rates'].append(float(drop_off))
                
                prev_retention = avg_retention
            else:
                # If day not available, use previous retention
                funnel_data['stages'].append(f'Day {day}')
                funnel_data['retention_rates'].append(float(prev_retention))
                funnel_data['drop_off_rates'].append(0.0)
        
        return funnel_data
    
    def analyze_seasonal_patterns(self, events_df: pd.DataFrame,
                                profiles: List[PlayerProfile]) -> Dict[str, Any]:
        """Analyze seasonal patterns in cohort performance."""
        
        cohort_table = self.create_cohort_table(events_df, profiles)
        
        seasonal_analysis = {
            'monthly_patterns': {},
            'day_of_week_patterns': {},
            'quarterly_trends': {}
        }
        
        # Analyze monthly patterns
        cohort_dates = pd.to_datetime(cohort_table.index)
        cohort_table_with_dates = cohort_table.copy()
        cohort_table_with_dates.index = cohort_dates
        
        # Group by month
        monthly_retention = cohort_table_with_dates.groupby(cohort_dates.month).mean()
        
        for month in monthly_retention.index:
            if 30 in monthly_retention.columns:
                seasonal_analysis['monthly_patterns'][int(month)] = {
                    'avg_30_day_retention': float(monthly_retention.loc[month, 30])
                }
        
        # Analyze day of week patterns
        dow_retention = cohort_table_with_dates.groupby(cohort_dates.dayofweek).mean()
        
        for dow in dow_retention.index:
            if 30 in dow_retention.columns:
                seasonal_analysis['day_of_week_patterns'][int(dow)] = {
                    'avg_30_day_retention': float(dow_retention.loc[dow, 30])
                }
        
        # Analyze quarterly trends
        quarterly_retention = cohort_table_with_dates.groupby(cohort_dates.quarter).mean()
        
        for quarter in quarterly_retention.index:
            if 30 in quarterly_retention.columns:
                seasonal_analysis['quarterly_trends'][int(quarter)] = {
                    'avg_30_day_retention': float(quarterly_retention.loc[quarter, 30])
                }
        
        return seasonal_analysis