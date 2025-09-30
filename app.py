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
    generate_sample_cohort_data,
    generate_sample_engagement_data,
    generate_sample_churn_data,
    generate_sample_funnel_data,
    generate_sample_level_data,
    generate_sample_cohort_funnel_data,
    create_churn_stats_summary,
    create_funnel_summary_stats
)

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

def get_sample_data():
    """Generate fresh sample data (simulates real-time data refresh)."""
    return {
        'cohort_data': generate_sample_cohort_data(),
        'engagement_data': generate_sample_engagement_data(),
        'churn_data': generate_sample_churn_data(),
        'funnel_data': generate_sample_funnel_data(),
        'level_data': generate_sample_level_data(),
        'cohort_funnel_data': generate_sample_cohort_funnel_data()
    }

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
    
    # Apply filters to data
    if filters['segment'] != 'all':
        churn_data = churn_data[churn_data['segment'] == filters['segment']]
    
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
                    start_date=filters['date_range'][0],
                    end_date=filters['date_range'][1],
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
    return create_main_layout()

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
    # Get fresh sample data
    data = get_sample_data()
    
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
    
    # Create tab content based on active tab
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
    
    return content, tabs

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
        
        data = get_sample_data()
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
        data = get_sample_data()
        
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
        data = get_sample_data()
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
        data = get_sample_data()
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
        data = get_sample_data()
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
        data = get_sample_data()
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
    app.run(debug=True, host='127.0.0.1', port=8050)