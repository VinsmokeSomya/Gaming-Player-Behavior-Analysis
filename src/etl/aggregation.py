"""
Aggregation functions for retention calculations and player metrics.
"""
import pandas as pd
from datetime import datetime, date, timedelta
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

from ..models import RetentionMetrics, PlayerProfile


class RetentionAggregator:
    """Handles aggregation of player data for retention analysis."""
    
    def __init__(self):
        """Initialize retention aggregator."""
        pass
    
    def calculate_daily_retention(self, events_df: pd.DataFrame, 
                                profiles: List[PlayerProfile]) -> List[RetentionMetrics]:
        """Calculate daily retention rates by registration cohort."""
        
        # Create profiles DataFrame for easier processing
        profiles_data = []
        for profile in profiles:
            profiles_data.append({
                'player_id': profile.player_id,
                'registration_date': profile.registration_date.date(),
                'segment': 'all'  # Default segment, can be enhanced later
            })
        
        profiles_df = pd.DataFrame(profiles_data)
        
        # Get daily active users from events
        events_df['date'] = pd.to_datetime(events_df['timestamp']).dt.date
        daily_active = events_df.groupby(['player_id', 'date']).size().reset_index(name='event_count')
        daily_active = daily_active[['player_id', 'date']].drop_duplicates()
        
        # Merge with registration dates
        player_activity = daily_active.merge(profiles_df, on='player_id', how='left')
        player_activity = player_activity.dropna()  # Remove players without profiles
        
        # Calculate retention for each cohort
        retention_metrics = []
        cohort_dates = sorted(profiles_df['registration_date'].unique())
        
        for cohort_date in cohort_dates:
            cohort_players = profiles_df[profiles_df['registration_date'] == cohort_date]['player_id'].tolist()
            cohort_size = len(cohort_players)
            
            if cohort_size == 0:
                continue
            
            # Calculate retention rates
            day_1_active = self._count_active_players_on_day(
                player_activity, cohort_players, cohort_date, 1
            )
            day_7_active = self._count_active_players_on_day(
                player_activity, cohort_players, cohort_date, 7
            )
            day_30_active = self._count_active_players_on_day(
                player_activity, cohort_players, cohort_date, 30
            )
            
            # Create retention metrics
            metrics = RetentionMetrics(
                cohort_date=cohort_date,
                day_1_retention=day_1_active / cohort_size if cohort_size > 0 else 0.0,
                day_7_retention=day_7_active / cohort_size if cohort_size > 0 else 0.0,
                day_30_retention=day_30_active / cohort_size if cohort_size > 0 else 0.0,
                cohort_size=cohort_size,
                segment='all'
            )
            
            retention_metrics.append(metrics)
        
        return retention_metrics
    
    def calculate_weekly_retention(self, events_df: pd.DataFrame,
                                 profiles: List[PlayerProfile]) -> List[RetentionMetrics]:
        """Calculate weekly retention rates by registration cohort."""
        
        # Group profiles by registration week
        profiles_data = []
        for profile in profiles:
            reg_date = profile.registration_date.date()
            # Get Monday of the registration week
            week_start = reg_date - timedelta(days=reg_date.weekday())
            profiles_data.append({
                'player_id': profile.player_id,
                'registration_week': week_start,
                'segment': 'all'
            })
        
        profiles_df = pd.DataFrame(profiles_data)
        
        # Get weekly active users
        events_df['date'] = pd.to_datetime(events_df['timestamp']).dt.date
        events_df['week_start'] = events_df['date'].apply(
            lambda x: x - timedelta(days=x.weekday())
        )
        
        weekly_active = events_df.groupby(['player_id', 'week_start']).size().reset_index(name='event_count')
        weekly_active = weekly_active[['player_id', 'week_start']].drop_duplicates()
        
        # Calculate retention for each weekly cohort
        retention_metrics = []
        cohort_weeks = sorted(profiles_df['registration_week'].unique())
        
        for cohort_week in cohort_weeks:
            cohort_players = profiles_df[
                profiles_df['registration_week'] == cohort_week
            ]['player_id'].tolist()
            cohort_size = len(cohort_players)
            
            if cohort_size == 0:
                continue
            
            # Calculate weekly retention rates
            week_1_active = self._count_active_players_in_week(
                weekly_active, cohort_players, cohort_week, 1
            )
            week_2_active = self._count_active_players_in_week(
                weekly_active, cohort_players, cohort_week, 2
            )
            week_4_active = self._count_active_players_in_week(
                weekly_active, cohort_players, cohort_week, 4
            )
            
            metrics = RetentionMetrics(
                cohort_date=cohort_week,
                day_1_retention=week_1_active / cohort_size if cohort_size > 0 else 0.0,
                day_7_retention=week_2_active / cohort_size if cohort_size > 0 else 0.0,
                day_30_retention=week_4_active / cohort_size if cohort_size > 0 else 0.0,
                cohort_size=cohort_size,
                segment='weekly'
            )
            
            retention_metrics.append(metrics)
        
        return retention_metrics
    
    def calculate_monthly_retention(self, events_df: pd.DataFrame,
                                  profiles: List[PlayerProfile]) -> List[RetentionMetrics]:
        """Calculate monthly retention rates by registration cohort."""
        
        # Group profiles by registration month
        profiles_data = []
        for profile in profiles:
            reg_date = profile.registration_date.date()
            month_start = reg_date.replace(day=1)
            profiles_data.append({
                'player_id': profile.player_id,
                'registration_month': month_start,
                'segment': 'all'
            })
        
        profiles_df = pd.DataFrame(profiles_data)
        
        # Get monthly active users
        events_df['date'] = pd.to_datetime(events_df['timestamp']).dt.date
        events_df['month_start'] = events_df['date'].apply(lambda x: x.replace(day=1))
        
        monthly_active = events_df.groupby(['player_id', 'month_start']).size().reset_index(name='event_count')
        monthly_active = monthly_active[['player_id', 'month_start']].drop_duplicates()
        
        # Calculate retention for each monthly cohort
        retention_metrics = []
        cohort_months = sorted(profiles_df['registration_month'].unique())
        
        for cohort_month in cohort_months:
            cohort_players = profiles_df[
                profiles_df['registration_month'] == cohort_month
            ]['player_id'].tolist()
            cohort_size = len(cohort_players)
            
            if cohort_size == 0:
                continue
            
            # Calculate monthly retention rates
            month_1_active = self._count_active_players_in_month(
                monthly_active, cohort_players, cohort_month, 1
            )
            month_2_active = self._count_active_players_in_month(
                monthly_active, cohort_players, cohort_month, 2
            )
            month_3_active = self._count_active_players_in_month(
                monthly_active, cohort_players, cohort_month, 3
            )
            
            metrics = RetentionMetrics(
                cohort_date=cohort_month,
                day_1_retention=month_1_active / cohort_size if cohort_size > 0 else 0.0,
                day_7_retention=month_2_active / cohort_size if cohort_size > 0 else 0.0,
                day_30_retention=month_3_active / cohort_size if cohort_size > 0 else 0.0,
                cohort_size=cohort_size,
                segment='monthly'
            )
            
            retention_metrics.append(metrics)
        
        return retention_metrics
    
    def _count_active_players_on_day(self, player_activity_df: pd.DataFrame,
                                   cohort_players: List[str], cohort_date: date,
                                   days_after: int) -> int:
        """Count active players on a specific day after cohort registration."""
        target_date = cohort_date + timedelta(days=days_after)
        
        active_on_date = player_activity_df[
            (player_activity_df['player_id'].isin(cohort_players)) &
            (player_activity_df['date'] == target_date)
        ]
        
        return len(active_on_date['player_id'].unique())
    
    def _count_active_players_in_week(self, weekly_activity_df: pd.DataFrame,
                                    cohort_players: List[str], cohort_week: date,
                                    weeks_after: int) -> int:
        """Count active players in a specific week after cohort registration."""
        target_week = cohort_week + timedelta(weeks=weeks_after)
        
        active_in_week = weekly_activity_df[
            (weekly_activity_df['player_id'].isin(cohort_players)) &
            (weekly_activity_df['week_start'] == target_week)
        ]
        
        return len(active_in_week['player_id'].unique())
    
    def _count_active_players_in_month(self, monthly_activity_df: pd.DataFrame,
                                     cohort_players: List[str], cohort_month: date,
                                     months_after: int) -> int:
        """Count active players in a specific month after cohort registration."""
        # Calculate target month
        year = cohort_month.year
        month = cohort_month.month + months_after
        
        # Handle year rollover
        while month > 12:
            month -= 12
            year += 1
        
        target_month = date(year, month, 1)
        
        active_in_month = monthly_activity_df[
            (monthly_activity_df['player_id'].isin(cohort_players)) &
            (monthly_activity_df['month_start'] == target_month)
        ]
        
        return len(active_in_month['player_id'].unique())
    
    def calculate_segment_retention(self, events_df: pd.DataFrame,
                                  profiles: List[PlayerProfile],
                                  segment_field: str = 'total_purchases') -> List[RetentionMetrics]:
        """Calculate retention rates by player segment."""
        
        # Create segments based on profile data
        profiles_data = []
        for profile in profiles:
            # Segment players based on spending
            if segment_field == 'total_purchases':
                if profile.total_purchases == 0:
                    segment = 'non_paying'
                elif profile.total_purchases < 10:
                    segment = 'low_spender'
                elif profile.total_purchases < 50:
                    segment = 'medium_spender'
                else:
                    segment = 'high_spender'
            else:
                segment = 'all'
            
            profiles_data.append({
                'player_id': profile.player_id,
                'registration_date': profile.registration_date.date(),
                'segment': segment
            })
        
        profiles_df = pd.DataFrame(profiles_data)
        
        # Calculate retention for each segment
        retention_metrics = []
        segments = profiles_df['segment'].unique()
        
        for segment in segments:
            segment_profiles = profiles_df[profiles_df['segment'] == segment]
            segment_retention = self._calculate_retention_for_segment(
                events_df, segment_profiles.to_dict('records'), segment
            )
            retention_metrics.extend(segment_retention)
        
        return retention_metrics
    
    def _calculate_retention_for_segment(self, events_df: pd.DataFrame,
                                       segment_profiles: List[Dict],
                                       segment_name: str) -> List[RetentionMetrics]:
        """Calculate retention metrics for a specific segment."""
        
        # Convert to profiles list format for reuse of existing logic
        from ..models import PlayerProfile
        profiles = []
        for profile_data in segment_profiles:
            # Create minimal profile for retention calculation
            profile = PlayerProfile(
                player_id=profile_data['player_id'],
                registration_date=datetime.combine(profile_data['registration_date'], datetime.min.time()),
                last_active_date=datetime.now(),
                total_sessions=0,
                total_playtime_minutes=0,
                highest_level_reached=1,
                total_purchases=0.0,
                churn_risk_score=0.0,
                churn_prediction_date=datetime.now()
            )
            profiles.append(profile)
        
        # Calculate daily retention for this segment
        retention_metrics = self.calculate_daily_retention(events_df, profiles)
        
        # Update segment name
        for metrics in retention_metrics:
            metrics.segment = segment_name
        
        return retention_metrics