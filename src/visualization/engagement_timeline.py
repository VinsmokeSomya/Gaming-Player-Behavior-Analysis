"""
Dynamic player engagement timeline charts with date range selection.
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from dash import dcc, html, callback, Input, Output
import dash_bootstrap_components as dbc
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta

from .components import ComponentFactory, ChartStyler, DashTheme


class EngagementTimelineGenerator:
    """Generates interactive player engagement timeline charts."""
    
    def __init__(self):
        self.chart_styler = ChartStyler()
    
    def create_engagement_timeline(
        self,
        engagement_data: pd.DataFrame,
        metrics: List[str] = None,
        title: str = "Player Engagement Timeline"
    ) -> go.Figure:
        """
        Create an interactive engagement timeline chart.
        
        Args:
            engagement_data: DataFrame with columns ['date', 'dau', 'wau', 'mau', 'avg_session_duration', 'sessions_per_user']
            metrics: List of metrics to display
            title: Chart title
            
        Returns:
            Plotly figure object
        """
        if metrics is None:
            metrics = ['dau', 'wau', 'mau']
        
        fig = go.Figure()
        
        # Color mapping for different metrics
        color_map = {
            'dau': DashTheme.PRIMARY_COLOR,
            'wau': DashTheme.SECONDARY_COLOR,
            'mau': DashTheme.SUCCESS_COLOR,
            'avg_session_duration': DashTheme.WARNING_COLOR,
            'sessions_per_user': DashTheme.INFO_COLOR
        }
        
        # Add traces for each metric
        for i, metric in enumerate(metrics):
            if metric in engagement_data.columns:
                # Format metric name for display
                metric_display = metric.replace('_', ' ').title()
                
                fig.add_trace(go.Scatter(
                    x=engagement_data['date'],
                    y=engagement_data[metric],
                    mode='lines+markers',
                    name=metric_display,
                    line=dict(
                        color=color_map.get(metric, DashTheme.CHART_COLORS[i % len(DashTheme.CHART_COLORS)]),
                        width=2
                    ),
                    marker=dict(size=6),
                    hovertemplate=f'<b>{metric_display}</b><br>' +
                                 'Date: %{x}<br>' +
                                 'Value: %{y:,.0f}<br>' +
                                 '<extra></extra>'
                ))
        
        # Apply base styling
        fig = self.chart_styler.apply_base_layout(fig, title)
        
        # Update layout for timeline
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Number of Players",
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Add range selector buttons
        fig.update_layout(
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=7, label="7D", step="day", stepmode="backward"),
                        dict(count=30, label="30D", step="day", stepmode="backward"),
                        dict(count=90, label="3M", step="day", stepmode="backward"),
                        dict(step="all", label="All")
                    ])
                ),
                rangeslider=dict(visible=True),
                type="date"
            )
        )
        
        return fig
    
    def create_session_metrics_chart(self, engagement_data: pd.DataFrame) -> go.Figure:
        """Create a chart showing session-related metrics."""
        fig = go.Figure()
        
        # Add average session duration
        fig.add_trace(go.Scatter(
            x=engagement_data['date'],
            y=engagement_data['avg_session_duration'],
            mode='lines+markers',
            name='Avg Session Duration (min)',
            yaxis='y',
            line=dict(color=DashTheme.WARNING_COLOR, width=2),
            hovertemplate='Date: %{x}<br>Duration: %{y:.1f} min<extra></extra>'
        ))
        
        # Add sessions per user on secondary y-axis
        fig.add_trace(go.Scatter(
            x=engagement_data['date'],
            y=engagement_data['sessions_per_user'],
            mode='lines+markers',
            name='Sessions per User',
            yaxis='y2',
            line=dict(color=DashTheme.INFO_COLOR, width=2),
            hovertemplate='Date: %{x}<br>Sessions: %{y:.2f}<extra></extra>'
        ))
        
        # Apply base styling
        fig = self.chart_styler.apply_base_layout(fig, "Session Metrics")
        
        # Update layout for dual y-axis
        fig.update_layout(
            yaxis=dict(
                title="Average Session Duration (minutes)",
                side="left"
            ),
            yaxis2=dict(
                title="Sessions per User",
                side="right",
                overlaying="y"
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    
    def create_engagement_distribution(self, engagement_data: pd.DataFrame) -> go.Figure:
        """Create a distribution chart of engagement levels."""
        # Calculate engagement score (simplified)
        engagement_data['engagement_score'] = (
            engagement_data['avg_session_duration'] * 
            engagement_data['sessions_per_user']
        )
        
        fig = go.Figure(data=go.Histogram(
            x=engagement_data['engagement_score'],
            nbinsx=30,
            marker_color=DashTheme.PRIMARY_COLOR,
            opacity=0.7,
            hovertemplate='Engagement Score: %{x:.1f}<br>Count: %{y}<extra></extra>'
        ))
        
        fig = self.chart_styler.apply_base_layout(fig, "Engagement Score Distribution")
        fig.update_layout(
            xaxis_title="Engagement Score",
            yaxis_title="Number of Days",
            height=300
        )
        
        return fig


def create_engagement_timeline_component(component_id: str = "engagement-timeline") -> html.Div:
    """
    Create the complete engagement timeline component with filters.
    
    Args:
        component_id: Base ID for the component
        
    Returns:
        Dash HTML component
    """
    return html.Div([
        # Filters row
        dbc.Row([
            dbc.Col([
                html.Label("Date Range:", className="fw-bold"),
                ComponentFactory.create_date_picker(
                    start_date=(datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
                    end_date=datetime.now().strftime('%Y-%m-%d'),
                    picker_id=f"{component_id}-date-picker"
                )
            ], width=4),
            dbc.Col([
                html.Label("Metrics to Display:", className="fw-bold"),
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
                    id=f"{component_id}-metrics-filter",
                    placeholder="Select metrics to display...",
                    style={"marginBottom": "10px"}
                )
            ], width=4),
            dbc.Col([
                html.Label("Player Segment:", className="fw-bold"),
                ComponentFactory.create_filter_dropdown(
                    options=[
                        {"label": "All Players", "value": "all"},
                        {"label": "New Players", "value": "new"},
                        {"label": "Returning Players", "value": "returning"},
                        {"label": "Premium Players", "value": "premium"}
                    ],
                    value="all",
                    dropdown_id=f"{component_id}-segment-filter",
                    placeholder="Select segment..."
                )
            ], width=4)
        ], className="mb-3"),
        
        # Main timeline chart
        ComponentFactory.create_loading_wrapper(
            dcc.Graph(
                id=f"{component_id}-graph",
                config={
                    'displayModeBar': True,
                    'displaylogo': False,
                    'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d']
                }
            ),
            f"{component_id}-loading"
        ),
        
        # Secondary charts row
        dbc.Row([
            dbc.Col([
                ComponentFactory.create_loading_wrapper(
                    dcc.Graph(
                        id=f"{component_id}-session-metrics",
                        config={'displayModeBar': False, 'displaylogo': False}
                    ),
                    f"{component_id}-session-loading"
                )
            ], width=8),
            dbc.Col([
                ComponentFactory.create_loading_wrapper(
                    dcc.Graph(
                        id=f"{component_id}-distribution",
                        config={'displayModeBar': False, 'displaylogo': False}
                    ),
                    f"{component_id}-distribution-loading"
                )
            ], width=4)
        ], className="mt-3")
    ])


# Sample data generation function removed - now using real data from ETL pipeline