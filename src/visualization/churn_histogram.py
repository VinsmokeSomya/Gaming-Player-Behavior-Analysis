"""
Interactive churn risk distribution histograms with segment filtering.
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from dash import dcc, html, callback, Input, Output
import dash_bootstrap_components as dbc
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import numpy as np

from .components import ComponentFactory, ChartStyler, DashTheme


class ChurnHistogramGenerator:
    """Generates interactive churn risk distribution histograms."""
    
    def __init__(self):
        self.chart_styler = ChartStyler()
    
    def create_churn_risk_histogram(
        self,
        churn_data: pd.DataFrame,
        segment_column: str = 'segment',
        title: str = "Churn Risk Distribution"
    ) -> go.Figure:
        """
        Create an interactive churn risk histogram.
        
        Args:
            churn_data: DataFrame with columns ['player_id', 'churn_risk_score', 'segment', 'days_since_last_session']
            segment_column: Column name for player segments
            title: Chart title
            
        Returns:
            Plotly figure object
        """
        fig = go.Figure()
        
        # Get unique segments
        segments = churn_data[segment_column].unique()
        colors = self.chart_styler.get_color_palette(len(segments))
        
        # Create histogram for each segment
        for i, segment in enumerate(segments):
            segment_data = churn_data[churn_data[segment_column] == segment]
            
            fig.add_trace(go.Histogram(
                x=segment_data['churn_risk_score'],
                name=segment,
                opacity=0.7,
                nbinsx=30,
                marker_color=colors[i],
                hovertemplate=f'<b>{segment}</b><br>' +
                             'Churn Risk: %{x:.2f}<br>' +
                             'Players: %{y}<br>' +
                             '<extra></extra>'
            ))
        
        # Apply base styling
        fig = self.chart_styler.apply_base_layout(fig, title)
        
        # Update layout for histogram
        fig.update_layout(
            xaxis_title="Churn Risk Score",
            yaxis_title="Number of Players",
            barmode='overlay',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Add risk threshold lines
        fig.add_vline(
            x=0.7, 
            line_dash="dash", 
            line_color="red",
            annotation_text="High Risk Threshold",
            annotation_position="top"
        )
        
        fig.add_vline(
            x=0.3, 
            line_dash="dash", 
            line_color="orange",
            annotation_text="Medium Risk Threshold",
            annotation_position="top"
        )
        
        return fig
    
    def create_risk_category_breakdown(self, churn_data: pd.DataFrame) -> go.Figure:
        """Create a pie chart showing risk category breakdown."""
        # Categorize risk levels
        def categorize_risk(score):
            if score >= 0.7:
                return "High Risk"
            elif score >= 0.3:
                return "Medium Risk"
            else:
                return "Low Risk"
        
        churn_data['risk_category'] = churn_data['churn_risk_score'].apply(categorize_risk)
        risk_counts = churn_data['risk_category'].value_counts()
        
        colors = {
            'High Risk': '#C73E1D',
            'Medium Risk': '#F18F01', 
            'Low Risk': '#2E86AB'
        }
        
        fig = go.Figure(data=go.Pie(
            labels=risk_counts.index,
            values=risk_counts.values,
            marker_colors=[colors[cat] for cat in risk_counts.index],
            hovertemplate='<b>%{label}</b><br>' +
                         'Players: %{value:,}<br>' +
                         'Percentage: %{percent}<br>' +
                         '<extra></extra>'
        ))
        
        fig = self.chart_styler.apply_base_layout(fig, "Risk Category Distribution")
        fig.update_layout(height=300)
        
        return fig
    
    def create_segment_risk_comparison(self, churn_data: pd.DataFrame) -> go.Figure:
        """Create a box plot comparing risk scores across segments."""
        fig = go.Figure()
        
        segments = churn_data['segment'].unique()
        colors = self.chart_styler.get_color_palette(len(segments))
        
        for i, segment in enumerate(segments):
            segment_data = churn_data[churn_data['segment'] == segment]
            
            fig.add_trace(go.Box(
                y=segment_data['churn_risk_score'],
                name=segment,
                marker_color=colors[i],
                boxpoints='outliers',
                hovertemplate=f'<b>{segment}</b><br>' +
                             'Risk Score: %{y:.3f}<br>' +
                             '<extra></extra>'
            ))
        
        fig = self.chart_styler.apply_base_layout(fig, "Risk Score by Segment")
        fig.update_layout(
            xaxis_title="Player Segment",
            yaxis_title="Churn Risk Score",
            height=350
        )
        
        return fig
    
    def create_risk_vs_engagement_scatter(self, churn_data: pd.DataFrame) -> go.Figure:
        """Create a scatter plot of risk score vs engagement metrics."""
        fig = go.Figure()
        
        # Create scatter plot
        fig.add_trace(go.Scatter(
            x=churn_data['days_since_last_session'],
            y=churn_data['churn_risk_score'],
            mode='markers',
            marker=dict(
                size=8,
                color=churn_data['churn_risk_score'],
                colorscale='RdYlBu_r',
                showscale=True,
                colorbar=dict(title="Churn Risk"),
                opacity=0.7
            ),
            text=churn_data['segment'],
            hovertemplate='Days Since Last Session: %{x}<br>' +
                         'Churn Risk: %{y:.3f}<br>' +
                         'Segment: %{text}<br>' +
                         '<extra></extra>'
        ))
        
        fig = self.chart_styler.apply_base_layout(fig, "Churn Risk vs Days Since Last Session")
        fig.update_layout(
            xaxis_title="Days Since Last Session",
            yaxis_title="Churn Risk Score",
            height=350
        )
        
        return fig


def create_churn_histogram_component(component_id: str = "churn-histogram") -> html.Div:
    """
    Create the complete churn histogram component with filters.
    
    Args:
        component_id: Base ID for the component
        
    Returns:
        Dash HTML component
    """
    return html.Div([
        # Filters row
        dbc.Row([
            dbc.Col([
                html.Label("Player Segment:", className="fw-bold"),
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
                    id=f"{component_id}-segment-filter",
                    placeholder="Select segments...",
                    style={"marginBottom": "10px"}
                )
            ], width=4),
            dbc.Col([
                html.Label("Risk Threshold:", className="fw-bold"),
                dcc.RangeSlider(
                    id=f"{component_id}-risk-slider",
                    min=0,
                    max=1,
                    step=0.1,
                    value=[0, 1],
                    marks={i/10: f'{i/10:.1f}' for i in range(0, 11, 2)},
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ], width=4),
            dbc.Col([
                html.Label("Days Since Last Session:", className="fw-bold"),
                dcc.RangeSlider(
                    id=f"{component_id}-days-slider",
                    min=0,
                    max=30,
                    step=1,
                    value=[0, 30],
                    marks={i: str(i) for i in range(0, 31, 5)},
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ], width=4)
        ], className="mb-3"),
        
        # Main histogram
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
                        id=f"{component_id}-pie-chart",
                        config={'displayModeBar': False, 'displaylogo': False}
                    ),
                    f"{component_id}-pie-loading"
                )
            ], width=4),
            dbc.Col([
                ComponentFactory.create_loading_wrapper(
                    dcc.Graph(
                        id=f"{component_id}-box-plot",
                        config={'displayModeBar': False, 'displaylogo': False}
                    ),
                    f"{component_id}-box-loading"
                )
            ], width=4),
            dbc.Col([
                ComponentFactory.create_loading_wrapper(
                    dcc.Graph(
                        id=f"{component_id}-scatter-plot",
                        config={'displayModeBar': False, 'displaylogo': False}
                    ),
                    f"{component_id}-scatter-loading"
                )
            ], width=4)
        ], className="mt-3"),
        
        # Summary statistics
        html.Div(id=f"{component_id}-stats", className="mt-3")
    ])


def create_churn_stats_summary(churn_data: pd.DataFrame) -> dbc.Row:
    """Create summary statistics cards for churn data."""
    total_players = len(churn_data)
    high_risk_players = len(churn_data[churn_data['churn_risk_score'] >= 0.7])
    avg_risk_score = churn_data['churn_risk_score'].mean()
    
    high_risk_pct = (high_risk_players / total_players) * 100 if total_players > 0 else 0
    
    return dbc.Row([
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
                f"{high_risk_pct:.1f}%"
            )
        ], width=3),
        dbc.Col([
            ComponentFactory.create_metric_card(
                "Average Risk Score",
                f"{avg_risk_score:.3f}",
                None
            )
        ], width=3),
        dbc.Col([
            ComponentFactory.create_metric_card(
                "Risk Categories",
                "3",
                "Low/Med/High"
            )
        ], width=3)
    ])


# Sample data generation function removed - now using real data from ETL pipeline