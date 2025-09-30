"""
Interactive cohort retention heatmap with hover details and zoom functionality.
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from dash import dcc, html, callback, Input, Output
import dash_bootstrap_components as dbc
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta

from .components import ComponentFactory, ChartStyler, DashTheme


class CohortHeatmapGenerator:
    """Generates interactive cohort retention heatmaps."""
    
    def __init__(self):
        self.chart_styler = ChartStyler()
    
    def create_cohort_heatmap(
        self, 
        cohort_data: pd.DataFrame,
        title: str = "Cohort Retention Heatmap",
        show_percentages: bool = True
    ) -> go.Figure:
        """
        Create an interactive cohort retention heatmap.
        
        Args:
            cohort_data: DataFrame with columns ['cohort_date', 'period', 'retention_rate', 'cohort_size']
            title: Chart title
            show_percentages: Whether to show percentages or raw numbers
            
        Returns:
            Plotly figure object
        """
        # Pivot data for heatmap format
        heatmap_data = cohort_data.pivot(
            index='cohort_date', 
            columns='period', 
            values='retention_rate'
        )
        
        # Create cohort size data for hover information
        cohort_sizes = cohort_data.pivot(
            index='cohort_date',
            columns='period', 
            values='cohort_size'
        )
        
        # Create hover text with detailed information
        hover_text = []
        for i, cohort_date in enumerate(heatmap_data.index):
            hover_row = []
            for j, period in enumerate(heatmap_data.columns):
                retention_rate = heatmap_data.iloc[i, j]
                cohort_size = cohort_sizes.iloc[i, j] if not pd.isna(cohort_sizes.iloc[i, j]) else 0
                
                if pd.isna(retention_rate):
                    hover_text_cell = "No data"
                else:
                    if show_percentages:
                        hover_text_cell = (
                            f"Cohort: {cohort_date}<br>"
                            f"Period: {period}<br>"
                            f"Retention: {retention_rate:.1%}<br>"
                            f"Players: {int(cohort_size):,}"
                        )
                    else:
                        hover_text_cell = (
                            f"Cohort: {cohort_date}<br>"
                            f"Period: {period}<br>"
                            f"Retained Players: {int(retention_rate * cohort_size):,}<br>"
                            f"Total Players: {int(cohort_size):,}"
                        )
                
                hover_row.append(hover_text_cell)
            hover_text.append(hover_row)
        
        # Create the heatmap
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=heatmap_data.columns,
            y=heatmap_data.index,
            hovertemplate='%{text}<extra></extra>',
            text=hover_text,
            colorscale='RdYlBu_r',
            zmin=0,
            zmax=1,
            colorbar=dict(
                title="Retention Rate",
                tickformat=".0%"
            )
        ))
        
        # Apply styling
        fig = self.chart_styler.apply_heatmap_styling(fig, title)
        
        # Update layout for better interactivity
        fig.update_layout(
            xaxis_title="Days Since Registration",
            yaxis_title="Cohort Registration Date",
            xaxis=dict(
                tickmode='linear',
                tick0=0,
                dtick=7  # Show every 7 days
            ),
            yaxis=dict(
                tickformat='%Y-%m-%d'
            )
        )
        
        return fig
    
    def create_cohort_size_chart(self, cohort_data: pd.DataFrame) -> go.Figure:
        """Create a supplementary chart showing cohort sizes."""
        cohort_sizes = cohort_data.groupby('cohort_date')['cohort_size'].first().reset_index()
        
        fig = go.Figure(data=go.Bar(
            x=cohort_sizes['cohort_date'],
            y=cohort_sizes['cohort_size'],
            marker_color=DashTheme.PRIMARY_COLOR,
            hovertemplate='Date: %{x}<br>Cohort Size: %{y:,}<extra></extra>'
        ))
        
        fig = self.chart_styler.apply_base_layout(fig, "Cohort Sizes")
        fig.update_layout(
            xaxis_title="Cohort Registration Date",
            yaxis_title="Number of Players",
            height=250
        )
        
        return fig


def create_cohort_heatmap_component(component_id: str = "cohort-heatmap") -> html.Div:
    """
    Create the complete cohort heatmap component with filters.
    
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
                    start_date=(datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d'),
                    end_date=datetime.now().strftime('%Y-%m-%d'),
                    picker_id=f"{component_id}-date-picker"
                )
            ], width=4),
            dbc.Col([
                html.Label("Segment:", className="fw-bold"),
                ComponentFactory.create_filter_dropdown(
                    options=[
                        {"label": "All Players", "value": "all"},
                        {"label": "Premium Players", "value": "premium"},
                        {"label": "Free Players", "value": "free"},
                        {"label": "High Engagement", "value": "high_engagement"},
                        {"label": "Low Engagement", "value": "low_engagement"}
                    ],
                    value="all",
                    dropdown_id=f"{component_id}-segment-filter",
                    placeholder="Select segment..."
                )
            ], width=4),
            dbc.Col([
                html.Label("Display Mode:", className="fw-bold"),
                ComponentFactory.create_filter_dropdown(
                    options=[
                        {"label": "Percentages", "value": "percentage"},
                        {"label": "Absolute Numbers", "value": "absolute"}
                    ],
                    value="percentage",
                    dropdown_id=f"{component_id}-display-mode",
                    placeholder="Select display mode..."
                )
            ], width=4)
        ], className="mb-3"),
        
        # Main heatmap
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
        
        # Cohort sizes chart
        ComponentFactory.create_loading_wrapper(
            dcc.Graph(
                id=f"{component_id}-sizes-graph",
                config={
                    'displayModeBar': False,
                    'displaylogo': False
                }
            ),
            f"{component_id}-sizes-loading"
        )
    ])


def generate_sample_cohort_data() -> pd.DataFrame:
    """Generate sample cohort data for testing."""
    import numpy as np
    
    # Generate sample data
    cohort_dates = pd.date_range(start='2024-01-01', end='2024-03-01', freq='W')
    periods = list(range(0, 91, 7))  # 0 to 90 days, weekly
    
    data = []
    for cohort_date in cohort_dates:
        base_size = np.random.randint(1000, 5000)
        for period in periods:
            # Simulate retention decay
            if period == 0:
                retention_rate = 1.0
                cohort_size = base_size
            else:
                # Exponential decay with some randomness
                decay_factor = np.exp(-period / 30)  # 30-day half-life
                retention_rate = decay_factor * np.random.uniform(0.8, 1.2)
                retention_rate = max(0.01, min(1.0, retention_rate))  # Clamp between 1% and 100%
                cohort_size = base_size
            
            data.append({
                'cohort_date': cohort_date.strftime('%Y-%m-%d'),
                'period': period,
                'retention_rate': retention_rate,
                'cohort_size': cohort_size
            })
    
    return pd.DataFrame(data)