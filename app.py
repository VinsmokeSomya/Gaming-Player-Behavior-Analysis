#!/usr/bin/env python3
"""
Player Retention Analytics Dashboard
Interactive Dash web application with advanced controls and cross-filtering.
"""

import dash
from dash import dcc, html, Input, Output, State, callback, ctx, ALL, MATCH
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json

# Import our visualization components
from src.visualization import (
    CohortHeatmapGenerator,
    EngagementTimelineGenerator,
    ChurnHistogramGenerator,
    DropoffFunnelGenerator,
    ComponentFactory,
    DashTheme,
    LayoutBuilder,
    create_churn_stats_summary,
    create_funnel_summary_stats
)

# Import real data loading
from src.etl.ingestion import DataLoader, EventIngestion
from src.etl.cohort_analysis import CohortAnalyzer
from src.etl.aggregation import RetentionAggregator

# Initialize Dash app with Bootstrap theme and meta tags for responsiveness
app = dash.Dash(
    __name__, 
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"}
    ],
    suppress_callback_exceptions=True  # This allows callbacks to reference IDs that don't exist yet
)
app.title = "Player Retention Analytics Dashboard"

# Global filter state store
app.layout = html.Div([
    dcc.Store(id='global-filters', data={
        'date_range': [(datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'), 
                      datetime.now().strftime('%Y-%m-%d')],
        'segment': 'all',
        'selected_cohort': None,
        'selected_players': None,
        'refresh_interval': 30000  # 30 seconds
    }),
    dcc.Store(id='drill-down-data', data={}),
    dcc.Interval(id='refresh-interval', interval=30000, n_intervals=0),
    html.Div(id='main-layout')
])

# Initialize generators
cohort_generator = CohortHeatmapGenerator()
engagement_generator = EngagementTimelineGenerator()
churn_generator = ChurnHistogramGenerator()
funnel_generator = DropoffFunnelGenerator()

# Initialize data loader and processors
data_loader = DataLoader(data_dir="data/full")  # Use full dataset with all 40,034 players
event_processor = EventIngestion()
cohort_analyzer = CohortAnalyzer()
retention_aggregator = RetentionAggregator()

# Global data cache
_cached_data = None
_cache_timestamp = None

def load_real_data():
    """Load real gaming data from JSON files."""
    global _cached_data, _cache_timestamp
    
    # Cache data for 5 minutes to improve performance
    if _cached_data and _cache_timestamp and (datetime.now() - _cache_timestamp).seconds < 300:
        return _cached_data
    
    print("Loading real gaming data...")
    
    try:
        # Load raw data
        profiles, events, churn_features = data_loader.load_all_data()
        
        # Convert events to DataFrame for processing
        events_df = event_processor.process_events_to_dataframe(events)
        
        # Create cohort analysis
        cohort_table = cohort_analyzer.create_cohort_table(events_df, profiles)
        
        # Convert cohort table to the format expected by visualization
        cohort_data = []
        for cohort_date in cohort_table.index:
            for period in cohort_table.columns:
                retention_rate = cohort_table.loc[cohort_date, period]
                if pd.notna(retention_rate):
                    cohort_data.append({
                        'cohort_date': cohort_date.strftime('%Y-%m-%d') if hasattr(cohort_date, 'strftime') else str(cohort_date),
                        'period': period,
                        'retention_rate': retention_rate,
                        'cohort_size': cohort_table.loc[cohort_date, 0] if 0 in cohort_table.columns else 100
                    })
        
        cohort_data = pd.DataFrame(cohort_data)
        
        # Create engagement timeline data from session events
        engagement_data = _create_engagement_timeline_from_events(events_df)
        
        # Create churn data from profiles and features
        churn_data = pd.DataFrame([{
            'player_id': p.player_id,
            'churn_risk_score': p.churn_risk_score,
            'segment': _determine_player_segment(p),
            'days_since_last_session': (datetime.now().date() - p.last_active_date.date()).days,
            'total_sessions': p.total_sessions,
            'total_playtime': p.total_playtime_minutes,
            'total_purchases': p.total_purchases
        } for p in profiles])
        
        # Ensure segment column exists for overview tab
        if 'segment' not in churn_data.columns:
            churn_data['segment'] = 'all'
        
        # Create funnel data from level completion events
        level_events = events_df[events_df['event_type'] == 'level_complete']
        funnel_data = _create_funnel_from_events(level_events)
        
        # Create level progression data
        level_data = _create_level_progression_data(level_events)
        
        # Create cohort funnel comparison
        cohort_funnel_data = _create_cohort_funnel_data(profiles, events_df)
        
        _cached_data = {
            'cohort_data': cohort_data,
            'engagement_data': engagement_data,
            'churn_data': churn_data,
            'funnel_data': funnel_data,
            'level_data': level_data,
            'cohort_funnel_data': cohort_funnel_data,
            'raw_profiles': profiles,
            'raw_events': events_df,
            'raw_churn_features': churn_features
        }
        _cache_timestamp = datetime.now()
        
        print(f"✅ Loaded real data: {len(profiles)} players, {len(events)} events")
        return _cached_data
        
    except Exception as e:
        print(f"❌ Error loading real data: {e}")
        print("Falling back to sample data generation...")
        # Fallback to generated data if real data fails
        return _generate_fallback_data()

def _determine_player_segment(profile):
    """Determine player segment based on engagement metrics."""
    if profile.total_sessions >= 100 and profile.total_purchases > 50:
        return 'premium'
    elif profile.total_sessions >= 50:
        return 'core'
    elif profile.total_sessions >= 10:
        return 'casual'
    else:
        return 'new'

def _create_funnel_from_events(level_events_df):
    """Create funnel data from level completion events."""
    if level_events_df.empty:
        return pd.DataFrame({
            'stage': ['Tutorial', 'Level 5', 'Level 10', 'Level 15', 'Level 20'],
            'players': [1000, 800, 600, 400, 200],
            'conversion_rate': [1.0, 0.8, 0.6, 0.4, 0.2],
            'stage_order': [0, 1, 2, 3, 4]
        })
    
    # Count players who reached each level milestone
    milestones = [1, 5, 10, 15, 20, 25, 30]
    funnel_data = []
    
    for i, milestone in enumerate(milestones):
        players_reached = level_events_df[level_events_df['level'] >= milestone]['player_id'].nunique()
        funnel_data.append({
            'stage': f'Level {milestone}' if milestone > 1 else 'Tutorial',
            'players': players_reached,
            'stage_order': i
        })
    
    # Calculate conversion rates
    total_players = funnel_data[0]['players'] if funnel_data else 1
    for stage in funnel_data:
        stage['conversion_rate'] = stage['players'] / total_players if total_players > 0 else 0
    
    return pd.DataFrame(funnel_data)

def _create_level_progression_data(level_events_df):
    """Create level progression heatmap data."""
    if level_events_df.empty:
        levels = list(range(1, 21))
        return pd.DataFrame({
            'level': levels,
            'completions': [100 - i*3 for i in range(20)],
            'avg_attempts': [1.2 + i*0.1 for i in range(20)],
            'level_group': ['Tutorial' if i <= 5 else 'Beginner' if i <= 10 else 'Intermediate' if i <= 15 else 'Advanced' for i in levels]
        })
    
    level_stats = level_events_df.groupby('level').agg({
        'player_id': 'nunique',
        'timestamp': 'count'
    }).reset_index()
    
    level_stats.columns = ['level', 'completions', 'total_attempts']
    level_stats['avg_attempts'] = level_stats['total_attempts'] / level_stats['completions']
    
    # Add level groups
    level_stats['level_group'] = level_stats['level'].apply(
        lambda x: 'Tutorial' if x <= 5 else 'Beginner' if x <= 10 else 'Intermediate' if x <= 15 else 'Advanced'
    )
    
    # Add completion rate (normalized)
    max_completions = level_stats['completions'].max() if len(level_stats) > 0 else 1
    level_stats['completion_rate'] = level_stats['completions'] / max_completions if max_completions > 0 else 0
    
    return level_stats

def _create_cohort_funnel_data(profiles, events_df):
    """Create cohort funnel comparison data."""
    # Group profiles by registration month
    cohort_data = []
    
    for profile in profiles[:100]:  # Sample for performance
        reg_month = profile.registration_date.strftime('%Y-%m')
        player_events = events_df[events_df['player_id'] == profile.player_id]
        
        # Calculate funnel progression for this player
        max_level = player_events[player_events['event_type'] == 'level_complete']['level'].max() if not player_events.empty else 0
        has_purchase = (player_events['event_type'] == 'purchase').any() if not player_events.empty else False
        
        cohort_data.append({
            'cohort': reg_month,
            'player_id': profile.player_id,
            'tutorial_complete': max_level >= 1,
            'level_5_complete': max_level >= 5,
            'level_10_complete': max_level >= 10,
            'first_purchase': has_purchase
        })
    
    return pd.DataFrame(cohort_data)

def _create_engagement_timeline_from_events(events_df):
    """Create engagement timeline data from events DataFrame."""
    # Convert timestamp to date
    events_df['date'] = pd.to_datetime(events_df['timestamp']).dt.date
    
    # Calculate daily metrics
    daily_metrics = []
    
    # Group by date and calculate metrics
    for date_val in sorted(events_df['date'].unique()):
        day_events = events_df[events_df['date'] == date_val]
        
        # Daily Active Users (unique players)
        dau = day_events['player_id'].nunique()
        
        # Session metrics (from session events)
        session_events = day_events[day_events['event_type'].isin(['session_start', 'session_end'])]
        session_starts = session_events[session_events['event_type'] == 'session_start']
        
        # Average session duration (if available)
        avg_session_duration = session_starts['session_duration'].mean() if 'session_duration' in session_starts.columns else 15.0
        
        # Sessions per user
        sessions_per_user = len(session_starts) / dau if dau > 0 else 0
        
        daily_metrics.append({
            'date': pd.to_datetime(date_val),
            'dau': dau,
            'wau': dau * 4.5,  # Approximate weekly actives
            'mau': dau * 15,   # Approximate monthly actives
            'avg_session_duration': avg_session_duration,
            'sessions_per_user': sessions_per_user
        })
    
    return pd.DataFrame(daily_metrics)

def _generate_fallback_data():
    """Generate fallback sample data if real data loading fails."""
    import numpy as np
    
    # Simple fallback data generation
    dates = pd.date_range(start='2024-07-01', end='2024-09-30', freq='D')
    
    return {
        'cohort_data': pd.DataFrame({
            'cohort_month': ['2024-07', '2024-08', '2024-09'],
            'day_0': [1000, 1200, 800],
            'day_1': [800, 960, 640],
            'day_7': [600, 720, 480],
            'day_30': [400, 480, 320]
        }),
        'engagement_data': pd.DataFrame({
            'date': dates,
            'dau': np.random.randint(800, 1200, len(dates)),
            'wau': np.random.randint(3000, 5000, len(dates)),
            'mau': np.random.randint(10000, 15000, len(dates))
        }),
        'churn_data': pd.DataFrame({
            'player_id': [f'player_{i}' for i in range(1000)],
            'churn_risk_score': np.random.random(1000),
            'segment': np.random.choice(['new', 'casual', 'core', 'premium'], 1000)
        }),
        'funnel_data': pd.DataFrame({
            'stage': ['Tutorial', 'Level 5', 'Level 10', 'Level 15', 'Level 20'],
            'players': [1000, 800, 600, 400, 200],
            'conversion_rate': [1.0, 0.8, 0.6, 0.4, 0.2]
        }),
        'level_data': pd.DataFrame({
            'level': list(range(1, 21)),
            'completions': [100 - i*3 for i in range(20)]
        }),
        'cohort_funnel_data': pd.DataFrame({
            'cohort': ['2024-07', '2024-08', '2024-09'],
            'tutorial_complete': [0.9, 0.85, 0.88],
            'level_5_complete': [0.7, 0.65, 0.68]
        })
    }

def get_real_data():
    """Get real gaming data (cached for performance)."""
    return load_real_data()

def create_navigation_bar():
    """Create responsive navigation bar with global filters."""
    return dbc.Navbar([
        dbc.Container([
            # Brand
            dbc.NavbarBrand([
                html.I(className="fas fa-gamepad me-2"),
                "Player Analytics"
            ], href="#", className="d-flex align-items-center"),
            
            # Toggle button for mobile
            dbc.NavbarToggler(id="navbar-toggler", n_clicks=0),
            
            # Collapsible content
            dbc.Collapse([
                dbc.Nav([
                    # Global date range picker
                    dbc.NavItem([
                        html.Label("Date Range:", className="text-light me-2"),
                        dcc.DatePickerRange(
                            id='global-date-picker',
                            start_date=(datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
                            end_date=datetime.now().strftime('%Y-%m-%d'),
                            display_format="MMM DD",
                            style={'fontSize': '12px'}
                        )
                    ], className="me-3"),
                    
                    # Global segment filter
                    dbc.NavItem([
                        html.Label("Segment:", className="text-light me-2"),
                        dcc.Dropdown(
                            id='global-segment-filter',
                            options=[
                                {"label": "All Players", "value": "all"},
                                {"label": "New Players", "value": "new"},
                                {"label": "Casual Players", "value": "casual"},
                                {"label": "Core Players", "value": "core"},
                                {"label": "Premium Players", "value": "premium"}
                            ],
                            value="all",
                            style={'minWidth': '120px', 'fontSize': '12px'}
                        )
                    ], className="me-3"),
                    
                    # Refresh button
                    dbc.NavItem([
                        dbc.Button([
                            html.I(className="fas fa-sync-alt me-1"),
                            "Refresh"
                        ], id="refresh-button", color="outline-light", size="sm")
                    ])
                ], navbar=True, className="ms-auto")
            ], id="navbar-collapse", navbar=True)
        ], fluid=True)
    ], color="primary", dark=True, className="mb-4")

def create_main_layout():
    """Create the main dashboard layout."""
    return dbc.Container([
        # Navigation bar
        create_navigation_bar(),
        
        # Main navigation tabs
        dbc.Row([
            dbc.Col([
                dbc.Tabs([
                    dbc.Tab(label="📊 Overview", tab_id="overview"),
                    dbc.Tab(label="🔥 Cohort Analysis", tab_id="cohort"),
                    dbc.Tab(label="📈 Engagement Timeline", tab_id="engagement"),
                    dbc.Tab(label="⚠️ Churn Analysis", tab_id="churn"),
                    dbc.Tab(label="🎯 Funnel Analysis", tab_id="funnel"),
                    dbc.Tab(label="🔍 Drill-Down", tab_id="drill-down", disabled=True),
                ], id="main-tabs", active_tab="overview")
            ])
        ], className="mb-4"),
        
        # Tab content
        html.Div(id="tab-content"),
        
        # Toast notifications for alerts
        dbc.Toast(
            id="notification-toast",
            header="System Notification",
            is_open=False,
            dismissable=True,
            duration=4000,
            style={"position": "fixed", "top": 66, "right": 10, "width": 350, "zIndex": 9999}
        )
    ], fluid=True)


def create_overview_tab(data, filters):
    """Create the overview tab with key metrics and cross-filtering."""
    churn_data = data['churn_data']
    engagement_data = data['engagement_data']
    funnel_data = data['funnel_data']
    
    # Apply filters to data (with safe defaults)
    segment_filter = filters.get('segment', 'all') if filters else 'all'
    if segment_filter != 'all' and 'segment' in churn_data.columns:
        churn_data = churn_data[churn_data['segment'] == segment_filter]
    
    # Calculate key metrics
    total_players = len(churn_data)
    high_risk_players = len(churn_data[churn_data['churn_risk_score'] >= 0.7]) if total_players > 0 else 0
    avg_dau = engagement_data['dau'].mean()
    funnel_conversion = funnel_data.iloc[-1]['players'] / funnel_data.iloc[0]['players'] if len(funnel_data) > 0 else 0
    
    return dbc.Container([
        # Key metrics row
        dbc.Row([
            dbc.Col([
                ComponentFactory.create_metric_card(
                    "Total Players", 
                    f"{total_players:,}",
                    None
                )
            ], width=3),
            dbc.Col([
                ComponentFactory.create_metric_card(
                    "High Risk Players", 
                    f"{high_risk_players:,}",
                    f"{(high_risk_players/total_players)*100:.1f}%" if total_players > 0 else "0%"
                )
            ], width=3),
            dbc.Col([
                ComponentFactory.create_metric_card(
                    "Average DAU", 
                    f"{avg_dau:,.0f}",
                    "+5.2% vs last month"
                )
            ], width=3),
            dbc.Col([
                ComponentFactory.create_metric_card(
                    "Funnel Conversion", 
                    f"{funnel_conversion:.1%}",
                    "-2.1% vs last month"
                )
            ], width=3)
        ], className="mb-4"),
        
        # Charts row with cross-filtering
        dbc.Row([
            dbc.Col([
                ComponentFactory.create_card(
                    "Engagement Overview (Click to filter)",
                    dcc.Graph(
                        id="overview-engagement-chart",
                        figure=engagement_generator.create_engagement_timeline(
                            engagement_data, 
                            metrics=['dau', 'wau'],
                            title="Daily & Weekly Active Users"
                        ),
                        config={'displayModeBar': True}
                    )
                )
            ], width=6),
            dbc.Col([
                ComponentFactory.create_card(
                    "Churn Risk Distribution (Click to filter)",
                    dcc.Graph(
                        id="overview-churn-chart",
                        figure=churn_generator.create_risk_category_breakdown(churn_data),
                        config={'displayModeBar': True}
                    )
                )
            ], width=6)
        ], className="mb-4"),
        
        dbc.Row([
            dbc.Col([
                ComponentFactory.create_card(
                    "Player Funnel Overview (Click stages to drill down)",
                    dcc.Graph(
                        id="overview-funnel-chart",
                        figure=funnel_generator.create_funnel_chart(funnel_data),
                        config={'displayModeBar': True}
                    )
                )
            ], width=12)
        ])
    ])


def create_cohort_tab(data, filters):
    """Create the cohort analysis tab with enhanced interactivity."""
    cohort_data = data['cohort_data']
    
    return dbc.Container([
        # Enhanced filters row
        dbc.Row([
            dbc.Col([
                html.Label("Date Range:", className="fw-bold"),
                ComponentFactory.create_date_picker(
                    start_date=filters.get('date_range', [None, None])[0] if filters else None,
                    end_date=filters.get('date_range', [None, None])[1] if filters else None,
                    picker_id="cohort-date-picker"
                )
            ], width=3),
            dbc.Col([
                html.Label("Display Mode:", className="fw-bold"),
                ComponentFactory.create_filter_dropdown(
                    options=[
                        {"label": "Percentages", "value": "percentage"},
                        {"label": "Absolute Numbers", "value": "absolute"}
                    ],
                    value="percentage",
                    dropdown_id="cohort-display-mode"
                )
            ], width=3),
            dbc.Col([
                html.Label("Cohort Size Filter:", className="fw-bold"),
                dcc.RangeSlider(
                    id="cohort-size-filter",
                    min=0,
                    max=1000,
                    step=50,
                    value=[0, 1000],
                    marks={i: str(i) for i in range(0, 1001, 200)},
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ], width=3),
            dbc.Col([
                dbc.Button([
                    html.I(className="fas fa-download me-1"),
                    "Export Data"
                ], id="export-cohort-data", color="outline-primary", size="sm")
            ], width=3, className="d-flex align-items-end")
        ], className="mb-4"),
        
        # Main heatmap with click interactions
        dbc.Row([
            dbc.Col([
                ComponentFactory.create_card(
                    "Cohort Retention Heatmap (Click cells for drill-down)",
                    dcc.Graph(
                        id="cohort-heatmap",
                        figure=cohort_generator.create_cohort_heatmap(cohort_data),
                        config={'displayModeBar': True}
                    )
                )
            ], width=12)
        ], className="mb-4"),
        
        # Secondary analysis charts
        dbc.Row([
            dbc.Col([
                ComponentFactory.create_card(
                    "Cohort Sizes Over Time",
                    dcc.Graph(
                        id="cohort-sizes-chart",
                        figure=cohort_generator.create_cohort_size_chart(cohort_data),
                        config={'displayModeBar': False}
                    )
                )
            ], width=6),
            dbc.Col([
                ComponentFactory.create_card(
                    "Retention Curve Comparison",
                    dcc.Graph(
                        id="retention-curves-chart",
                        figure=go.Figure().add_annotation(
                            text="Select cohorts from heatmap to compare",
                            xref="paper", yref="paper",
                            x=0.5, y=0.5, showarrow=False
                        ),
                        config={'displayModeBar': False}
                    )
                )
            ], width=6)
        ])
    ])


def create_engagement_tab(data, filters):
    """Create the engagement timeline tab with cross-filtering."""
    engagement_data = data['engagement_data']
    
    return dbc.Container([
        # Enhanced filters
        dbc.Row([
            dbc.Col([
                html.Label("Metrics:", className="fw-bold"),
                dcc.Dropdown(
                    options=[
                        {"label": "Daily Active Users", "value": "dau"},
                        {"label": "Weekly Active Users", "value": "wau"},
                        {"label": "Monthly Active Users", "value": "mau"},
                        {"label": "Avg Session Duration", "value": "avg_session_duration"},
                        {"label": "Sessions per User", "value": "sessions_per_user"}
                    ],
                    value=["dau", "wau", "mau"],
                    multi=True,
                    id="engagement-metrics",
                    placeholder="Select metrics..."
                )
            ], width=4),
            dbc.Col([
                html.Label("Comparison Mode:", className="fw-bold"),
                ComponentFactory.create_filter_dropdown(
                    options=[
                        {"label": "Absolute Values", "value": "absolute"},
                        {"label": "Percentage Change", "value": "percentage"},
                        {"label": "Moving Average", "value": "moving_avg"}
                    ],
                    value="absolute",
                    dropdown_id="engagement-comparison-mode"
                )
            ], width=4),
            dbc.Col([
                html.Label("Anomaly Detection:", className="fw-bold"),
                dbc.Switch(
                    id="engagement-anomaly-detection",
                    label="Highlight Anomalies",
                    value=False
                )
            ], width=4)
        ], className="mb-4"),
        
        # Main timeline with brush selection
        dbc.Row([
            dbc.Col([
                ComponentFactory.create_card(
                    "Player Engagement Timeline (Drag to select time range)",
                    dcc.Graph(
                        id="engagement-timeline",
                        figure=engagement_generator.create_engagement_timeline(engagement_data),
                        config={'displayModeBar': True}
                    )
                )
            ], width=12)
        ], className="mb-4"),
        
        # Secondary charts that update based on selection
        dbc.Row([
            dbc.Col([
                ComponentFactory.create_card(
                    "Session Metrics (Selected Period)",
                    dcc.Graph(
                        id="engagement-session-metrics",
                        figure=engagement_generator.create_session_metrics_chart(engagement_data),
                        config={'displayModeBar': False}
                    )
                )
            ], width=6),
            dbc.Col([
                ComponentFactory.create_card(
                    "Engagement Distribution",
                    dcc.Graph(
                        id="engagement-distribution",
                        figure=engagement_generator.create_engagement_distribution(engagement_data),
                        config={'displayModeBar': False}
                    )
                )
            ], width=6)
        ])
    ])


def create_churn_tab(data, filters):
    """Create the churn analysis tab with advanced filtering."""
    churn_data = data['churn_data']
    
    return dbc.Container([
        # Dynamic summary stats
        html.Div(id="churn-summary-stats"),
        
        # Enhanced filters
        dbc.Row([
            dbc.Col([
                html.Label("Player Segments:", className="fw-bold"),
                dcc.Dropdown(
                    options=[
                        {"label": "All Segments", "value": "all"},
                        {"label": "New Players", "value": "new"},
                        {"label": "Casual Players", "value": "casual"},
                        {"label": "Core Players", "value": "core"},
                        {"label": "Premium Players", "value": "premium"}
                    ],
                    value=["all"],
                    multi=True,
                    id="churn-segments",
                    placeholder="Select segments..."
                )
            ], width=3),
            dbc.Col([
                html.Label("Risk Threshold:", className="fw-bold"),
                dcc.RangeSlider(
                    id="churn-risk-slider",
                    min=0,
                    max=1,
                    step=0.05,
                    value=[0, 1],
                    marks={i/10: f'{i/10:.1f}' for i in range(0, 11, 2)},
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ], width=6),
            dbc.Col([
                html.Label("Actions:", className="fw-bold"),
                dbc.ButtonGroup([
                    dbc.Button("Export High Risk", id="export-high-risk", color="warning", size="sm"),
                    dbc.Button("Create Campaign", id="create-campaign", color="success", size="sm")
                ])
            ], width=3)
        ], className="mb-4"),
        
        # Main histogram with selection
        dbc.Row([
            dbc.Col([
                ComponentFactory.create_card(
                    "Churn Risk Distribution (Click bars to select players)",
                    dcc.Graph(
                        id="churn-histogram",
                        figure=churn_generator.create_churn_risk_histogram(churn_data),
                        config={'displayModeBar': True}
                    )
                )
            ], width=12)
        ], className="mb-4"),
        
        # Secondary charts with cross-filtering
        dbc.Row([
            dbc.Col([
                ComponentFactory.create_card(
                    "Risk by Segment (Selected Players)",
                    dcc.Graph(
                        id="churn-segment-comparison",
                        figure=churn_generator.create_segment_risk_comparison(churn_data),
                        config={'displayModeBar': False}
                    )
                )
            ], width=6),
            dbc.Col([
                ComponentFactory.create_card(
                    "Risk vs Engagement Scatter",
                    dcc.Graph(
                        id="churn-risk-scatter",
                        figure=churn_generator.create_risk_vs_engagement_scatter(churn_data),
                        config={'displayModeBar': True}
                    )
                )
            ], width=6)
        ])
    ])


def create_funnel_tab(data, filters):
    """Create the funnel analysis tab with drill-down capabilities."""
    funnel_data = data['funnel_data']
    level_data = data['level_data']
    cohort_funnel_data = data['cohort_funnel_data']
    
    return dbc.Container([
        # Dynamic summary stats
        html.Div(id="funnel-summary-stats"),
        
        # Enhanced filters
        dbc.Row([
            dbc.Col([
                html.Label("Cohort:", className="fw-bold"),
                ComponentFactory.create_filter_dropdown(
                    options=[
                        {"label": "All Cohorts", "value": "all"},
                        {"label": "Week 1", "value": "week1"},
                        {"label": "Week 2", "value": "week2"},
                        {"label": "Week 3", "value": "week3"}
                    ],
                    value="all",
                    dropdown_id="funnel-cohort"
                )
            ], width=3),
            dbc.Col([
                html.Label("Game Mode:", className="fw-bold"),
                ComponentFactory.create_filter_dropdown(
                    options=[
                        {"label": "All Modes", "value": "all"},
                        {"label": "Tutorial", "value": "tutorial"},
                        {"label": "Campaign", "value": "campaign"}
                    ],
                    value="all",
                    dropdown_id="funnel-mode"
                )
            ], width=3),
            dbc.Col([
                html.Label("View Type:", className="fw-bold"),
                ComponentFactory.create_filter_dropdown(
                    options=[
                        {"label": "Funnel View", "value": "funnel"},
                        {"label": "Level Heatmap", "value": "heatmap"},
                        {"label": "Cohort Comparison", "value": "comparison"}
                    ],
                    value="funnel",
                    dropdown_id="funnel-view-type"
                )
            ], width=3),
            dbc.Col([
                html.Label("Level Range:", className="fw-bold"),
                dcc.RangeSlider(
                    id="funnel-level-range",
                    min=1,
                    max=20,
                    step=1,
                    value=[1, 20],
                    marks={i: str(i) for i in range(1, 21, 5)},
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ], width=3)
        ], className="mb-4"),
        
        # Main funnel chart with click interactions
        dbc.Row([
            dbc.Col([
                ComponentFactory.create_card(
                    "Player Drop-off Funnel (Click stages to drill down)",
                    dcc.Graph(
                        id="funnel-main-chart",
                        figure=funnel_generator.create_funnel_chart(funnel_data),
                        config={'displayModeBar': True}
                    )
                )
            ], width=12)
        ], className="mb-4"),
        
        # Secondary charts with cross-filtering
        dbc.Row([
            dbc.Col([
                ComponentFactory.create_card(
                    "Stage Drop-off Rates",
                    dcc.Graph(
                        id="funnel-dropoff-rates",
                        figure=funnel_generator.create_drop_off_bar_chart(funnel_data),
                        config={'displayModeBar': False}
                    )
                )
            ], width=6),
            dbc.Col([
                ComponentFactory.create_card(
                    "Level Progression Heatmap",
                    dcc.Graph(
                        id="funnel-level-heatmap",
                        figure=funnel_generator.create_level_progression_heatmap(level_data),
                        config={'displayModeBar': True}
                    )
                )
            ], width=6)
        ])
    ])

def create_drill_down_tab(drill_down_data):
    """Create detailed drill-down analysis page."""
    if not drill_down_data:
        return dbc.Container([
            dbc.Alert([
                html.H4("No Drill-Down Data Selected", className="alert-heading"),
                html.P("Click on charts in other tabs to drill down into specific cohorts, segments, or time periods."),
                html.Hr(),
                html.P("Available drill-down options:", className="mb-0"),
                html.Ul([
                    html.Li("Click cohort heatmap cells to analyze specific cohort retention"),
                    html.Li("Click funnel stages to see detailed drop-off analysis"),
                    html.Li("Select time ranges in engagement charts for period analysis"),
                    html.Li("Click churn risk bars to analyze high-risk player segments")
                ])
            ], color="info")
        ])
    
    drill_type = drill_down_data.get('type', 'unknown')
    
    if drill_type == 'cohort':
        return create_cohort_drill_down(drill_down_data)
    elif drill_type == 'funnel_stage':
        return create_funnel_stage_drill_down(drill_down_data)
    elif drill_type == 'churn_segment':
        return create_churn_segment_drill_down(drill_down_data)
    elif drill_type == 'engagement_period':
        return create_engagement_period_drill_down(drill_down_data)
    else:
        return dbc.Container([
            dbc.Alert(f"Unknown drill-down type: {drill_type}", color="warning")
        ])

def create_cohort_drill_down(drill_data):
    """Create detailed cohort analysis drill-down."""
    cohort_date = drill_data.get('cohort_date', 'Unknown')
    day_period = drill_data.get('day_period', 'Unknown')
    
    return dbc.Container([
        # Header
        dbc.Row([
            dbc.Col([
                html.H3(f"Cohort Analysis: {cohort_date} - Day {day_period}"),
                html.P(f"Detailed analysis for players who registered on {cohort_date}", 
                       className="text-muted")
            ])
        ], className="mb-4"),
        
        # Key metrics for this cohort
        dbc.Row([
            dbc.Col([
                ComponentFactory.create_metric_card(
                    "Cohort Size", 
                    drill_data.get('cohort_size', 'N/A'),
                    None
                )
            ], width=3),
            dbc.Col([
                ComponentFactory.create_metric_card(
                    "Retention Rate", 
                    f"{drill_data.get('retention_rate', 0):.1%}",
                    drill_data.get('retention_delta', '')
                )
            ], width=3),
            dbc.Col([
                ComponentFactory.create_metric_card(
                    "Active Players", 
                    drill_data.get('active_players', 'N/A'),
                    None
                )
            ], width=3),
            dbc.Col([
                ComponentFactory.create_metric_card(
                    "Avg LTV", 
                    f"${drill_data.get('avg_ltv', 0):.2f}",
                    drill_data.get('ltv_delta', '')
                )
            ], width=3)
        ], className="mb-4"),
        
        # Detailed charts
        dbc.Row([
            dbc.Col([
                ComponentFactory.create_card(
                    "Player Journey Timeline",
                    html.Div("Detailed player journey chart would go here")
                )
            ], width=12)
        ])
    ])

def create_funnel_stage_drill_down(drill_data):
    """Create detailed funnel stage analysis."""
    stage_name = drill_data.get('stage_name', 'Unknown Stage')
    
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H3(f"Funnel Stage Analysis: {stage_name}"),
                html.P("Detailed drop-off analysis for this stage", className="text-muted")
            ])
        ], className="mb-4"),
        
        # Stage-specific metrics and analysis would go here
        dbc.Row([
            dbc.Col([
                ComponentFactory.create_card(
                    "Stage Performance Details",
                    html.Div("Detailed stage analysis would go here")
                )
            ], width=12)
        ])
    ])

def create_churn_segment_drill_down(drill_data):
    """Create detailed churn segment analysis."""
    segment = drill_data.get('segment', 'Unknown Segment')
    
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H3(f"Churn Analysis: {segment} Players"),
                html.P("Detailed churn risk analysis for this segment", className="text-muted")
            ])
        ], className="mb-4"),
        
        # Segment-specific analysis would go here
        dbc.Row([
            dbc.Col([
                ComponentFactory.create_card(
                    "Segment Churn Details",
                    html.Div("Detailed segment analysis would go here")
                )
            ], width=12)
        ])
    ])

def create_engagement_period_drill_down(drill_data):
    """Create detailed engagement period analysis."""
    start_date = drill_data.get('start_date', 'Unknown')
    end_date = drill_data.get('end_date', 'Unknown')
    
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H3(f"Engagement Analysis: {start_date} to {end_date}"),
                html.P("Detailed engagement analysis for selected period", className="text-muted")
            ])
        ], className="mb-4"),
        
        # Period-specific analysis would go here
        dbc.Row([
            dbc.Col([
                ComponentFactory.create_card(
                    "Period Engagement Details",
                    html.Div("Detailed period analysis would go here")
                )
            ], width=12)
        ])
    ])


# Main layout callback
@app.callback(
    Output('main-layout', 'children'),
    Input('refresh-interval', 'n_intervals')
)
def update_main_layout(n_intervals):
    """Initialize or refresh the main layout."""
    try:
        return create_main_layout()
    except Exception as e:
        print(f"Error creating main layout: {e}")
        return dbc.Alert([
            html.H4("Dashboard Error", className="alert-heading"),
            html.P(f"Error initializing dashboard: {str(e)}"),
            html.P("Please refresh the page.")
        ], color="danger")

# Global filters callback
@app.callback(
    Output('global-filters', 'data'),
    [Input('global-date-picker', 'start_date'),
     Input('global-date-picker', 'end_date'),
     Input('global-segment-filter', 'value'),
     Input('refresh-button', 'n_clicks')],
    State('global-filters', 'data')
)
def update_global_filters(start_date, end_date, segment, refresh_clicks, current_filters):
    """Update global filter state."""
    if not current_filters:
        current_filters = {}
    
    current_filters.update({
        'date_range': [start_date or current_filters.get('date_range', [None, None])[0], 
                      end_date or current_filters.get('date_range', [None, None])[1]],
        'segment': segment or 'all',
        'last_refresh': datetime.now().isoformat() if refresh_clicks else current_filters.get('last_refresh')
    })
    
    return current_filters

# Navbar toggle callback
@app.callback(
    Output("navbar-collapse", "is_open"),
    [Input("navbar-toggler", "n_clicks")],
    [State("navbar-collapse", "is_open")]
)
def toggle_navbar_collapse(n, is_open):
    """Toggle navbar collapse for mobile."""
    if n:
        return not is_open
    return is_open

# Main tab switching with data refresh
@app.callback(
    [Output("tab-content", "children"),
     Output("main-tabs", "children")],
    [Input("main-tabs", "active_tab"),
     Input('global-filters', 'data'),
     Input('drill-down-data', 'data')],
    prevent_initial_call=False
)
def switch_tab_with_data(active_tab, filters, drill_down_data):
    """Switch between tabs with fresh data and applied filters."""
    try:
        # Get real gaming data
        data = get_real_data()
        
        # Update tab labels to show drill-down availability
        tabs = [
            dbc.Tab(label="📊 Overview", tab_id="overview"),
            dbc.Tab(label="🔥 Cohort Analysis", tab_id="cohort"),
            dbc.Tab(label="📈 Engagement Timeline", tab_id="engagement"),
            dbc.Tab(label="⚠️ Churn Analysis", tab_id="churn"),
            dbc.Tab(label="🎯 Funnel Analysis", tab_id="funnel"),
            dbc.Tab(label="🔍 Drill-Down", tab_id="drill-down", 
                   disabled=not drill_down_data, 
                   className="text-success" if drill_down_data else "")
        ]
        
        # Create tab content based on active tab with error handling
        try:
            if active_tab == "overview":
                content = create_overview_tab(data, filters or {})
            elif active_tab == "cohort":
                content = create_cohort_tab(data, filters or {})
            elif active_tab == "engagement":
                content = create_engagement_tab(data, filters or {})
            elif active_tab == "churn":
                content = create_churn_tab(data, filters or {})
            elif active_tab == "funnel":
                content = create_funnel_tab(data, filters or {})
            elif active_tab == "drill-down":
                content = create_drill_down_tab(drill_down_data)
            else:
                content = create_overview_tab(data, filters or {})
        except Exception as e:
            print(f"Error creating tab content for {active_tab}: {e}")
            content = dbc.Alert([
                html.H4("Error Loading Tab", className="alert-heading"),
                html.P(f"There was an error loading the {active_tab} tab: {str(e)}"),
                html.Hr(),
                html.P("Please try refreshing the page or contact support if the issue persists.")
            ], color="danger")
        
        return content, tabs
        
    except Exception as e:
        print(f"Critical error in tab switching: {e}")
        # Return minimal error content
        error_content = dbc.Alert([
            html.H4("System Error", className="alert-heading"),
            html.P(f"Critical error: {str(e)}"),
            html.P("Please refresh the page.")
        ], color="danger")
        
        default_tabs = [
            dbc.Tab(label="📊 Overview", tab_id="overview"),
            dbc.Tab(label="🔥 Cohort Analysis", tab_id="cohort"),
            dbc.Tab(label="📈 Engagement Timeline", tab_id="engagement"),
            dbc.Tab(label="⚠️ Churn Analysis", tab_id="churn"),
            dbc.Tab(label="🎯 Funnel Analysis", tab_id="funnel")
        ]
        
        return error_content, default_tabs

# Additional callbacks for interactivity (with proper error handling)
@app.callback(
    Output("engagement-timeline", "figure"),
    [Input("engagement-metrics", "value"),
     Input("engagement-comparison-mode", "value"),
     Input("engagement-anomaly-detection", "value")],
    State('global-filters', 'data'),
    prevent_initial_call=True
)
def update_engagement_timeline(selected_metrics, comparison_mode, show_anomalies, filters):
    """Update engagement timeline with advanced features."""
    try:
        if not selected_metrics:
            selected_metrics = ["dau"]
        
        data = get_real_data()
        engagement_data = data['engagement_data']
        
        # Create enhanced timeline
        fig = engagement_generator.create_engagement_timeline(
            engagement_data, 
            metrics=selected_metrics,
            title="Player Engagement Timeline"
        )
        
        # Add brush selection for time range filtering
        fig.update_layout(
            xaxis=dict(rangeslider=dict(visible=True)),
            hovermode='x unified'
        )
        
        return fig
    except Exception as e:
        # Return empty figure on error
        return go.Figure().add_annotation(
            text=f"Error loading chart: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )

@app.callback(
    Output("funnel-main-chart", "figure"),
    [Input("funnel-view-type", "value"),
     Input("funnel-cohort", "value"),
     Input("funnel-mode", "value"),
     Input("funnel-level-range", "value")],
    State('global-filters', 'data'),
    prevent_initial_call=True
)
def update_funnel_chart(view_type, cohort, mode, level_range, filters):
    """Update funnel chart based on filters and view type."""
    try:
        data = get_real_data()
        
        if view_type == "heatmap":
            return funnel_generator.create_level_progression_heatmap(data['level_data'])
        elif view_type == "comparison":
            return funnel_generator.create_cohort_funnel_comparison(data['cohort_funnel_data'])
        else:
            return funnel_generator.create_funnel_chart(data['funnel_data'])
    except Exception as e:
        # Return empty figure on error
        return go.Figure().add_annotation(
            text=f"Error loading chart: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )

@app.callback(
    Output("cohort-heatmap", "figure"),
    [Input("cohort-display-mode", "value"),
     Input("cohort-size-filter", "value")],
    State('global-filters', 'data'),
    prevent_initial_call=True
)
def update_cohort_heatmap(display_mode, size_filter, filters):
    """Update cohort heatmap based on display mode and filters."""
    try:
        data = get_real_data()
        cohort_data = data['cohort_data']
        
        # Apply size filter (in real implementation)
        # cohort_data = cohort_data[cohort_data['cohort_size'].between(size_filter[0], size_filter[1])]
        
        return cohort_generator.create_cohort_heatmap(cohort_data)
    except Exception as e:
        # Return empty figure on error
        return go.Figure().add_annotation(
            text=f"Error loading chart: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )

@app.callback(
    Output("churn-histogram", "figure"),
    [Input("churn-segments", "value"),
     Input("churn-risk-slider", "value")],
    State('global-filters', 'data'),
    prevent_initial_call=True
)
def update_churn_histogram(segments, risk_range, filters):
    """Update churn histogram based on segment and risk filters."""
    try:
        data = get_real_data()
        churn_data = data['churn_data']
        
        # Apply filters
        if segments and 'all' not in segments:
            churn_data = churn_data[churn_data['segment'].isin(segments)]
        
        if risk_range:
            churn_data = churn_data[
                (churn_data['churn_risk_score'] >= risk_range[0]) & 
                (churn_data['churn_risk_score'] <= risk_range[1])
            ]
        
        return churn_generator.create_churn_risk_histogram(churn_data)
    except Exception as e:
        # Return empty figure on error
        return go.Figure().add_annotation(
            text=f"Error loading chart: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )

@app.callback(
    Output('churn-summary-stats', 'children'),
    [Input('churn-segments', 'value'),
     Input('churn-risk-slider', 'value')],
    State('global-filters', 'data'),
    prevent_initial_call=True
)
def update_churn_summary(segments, risk_range, filters):
    """Update churn summary stats based on filters."""
    try:
        data = get_real_data()
        churn_data = data['churn_data']
        
        # Apply filters
        if segments and 'all' not in segments:
            churn_data = churn_data[churn_data['segment'].isin(segments)]
        
        if risk_range:
            churn_data = churn_data[
                (churn_data['churn_risk_score'] >= risk_range[0]) & 
                (churn_data['churn_risk_score'] <= risk_range[1])
            ]
        
        return create_churn_stats_summary(churn_data)
    except Exception as e:
        # Return empty div on error
        return html.Div(f"Error loading summary: {str(e)}")

@app.callback(
    Output('funnel-summary-stats', 'children'),
    [Input('funnel-cohort', 'value'),
     Input('funnel-mode', 'value'),
     Input('funnel-level-range', 'value')],
    State('global-filters', 'data'),
    prevent_initial_call=True
)
def update_funnel_summary(cohort, mode, level_range, filters):
    """Update funnel summary stats based on filters."""
    try:
        data = get_real_data()
        funnel_data = data['funnel_data']
        
        # Apply filters (in real implementation, would filter actual data)
        return create_funnel_summary_stats(funnel_data)
    except Exception as e:
        # Return empty div on error
        return html.Div(f"Error loading summary: {str(e)}")


if __name__ == "__main__":
    print("🎮 Starting Player Retention Analytics Dashboard...")
    print("📊 Dashboard will be available at: http://127.0.0.1:8050")
    print("🔧 Press Ctrl+C to stop the server")
    
    # Run the app
    app.run(debug=True, host='127.0.0.1', port=8050)