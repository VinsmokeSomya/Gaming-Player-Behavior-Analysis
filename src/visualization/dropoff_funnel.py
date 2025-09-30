"""
Drop-off funnel visualization with clickable level drill-down functionality.
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


class DropoffFunnelGenerator:
    """Generates interactive drop-off funnel visualizations."""
    
    def __init__(self):
        self.chart_styler = ChartStyler()
    
    def create_funnel_chart(
        self,
        funnel_data: pd.DataFrame,
        title: str = "Player Drop-off Funnel",
        show_percentages: bool = True
    ) -> go.Figure:
        """
        Create an interactive funnel chart showing player drop-off.
        
        Args:
            funnel_data: DataFrame with columns ['stage', 'players', 'stage_order']
            title: Chart title
            show_percentages: Whether to show percentages or absolute numbers
            
        Returns:
            Plotly figure object
        """
        # Sort by stage order
        funnel_data = funnel_data.sort_values('stage_order')
        
        # Calculate conversion rates
        total_players = funnel_data.iloc[0]['players']
        funnel_data['conversion_rate'] = funnel_data['players'] / total_players
        funnel_data['drop_off_rate'] = 1 - funnel_data['conversion_rate']
        
        # Calculate stage-to-stage drop-off
        funnel_data['stage_drop_off'] = 0.0  # Use float instead of int
        for i in range(1, len(funnel_data)):
            prev_players = funnel_data.iloc[i-1]['players']
            curr_players = funnel_data.iloc[i]['players']
            drop_off_rate = (prev_players - curr_players) / prev_players if prev_players > 0 else 0.0
            funnel_data.at[funnel_data.index[i], 'stage_drop_off'] = drop_off_rate
        
        # Create funnel chart
        fig = go.Figure()
        
        # Add funnel trace
        fig.add_trace(go.Funnel(
            y=funnel_data['stage'],
            x=funnel_data['players'],
            textinfo="value+percent initial",
            texttemplate='%{value:,}<br>(%{percentInitial})',
            marker=dict(
                color=DashTheme.CHART_COLORS[:len(funnel_data)],
                line=dict(width=2, color="white")
            ),
            hovertemplate='<b>%{y}</b><br>' +
                         'Players: %{value:,}<br>' +
                         'Conversion: %{percentInitial}<br>' +
                         '<extra></extra>',
            connector=dict(
                line=dict(color="lightgray", dash="dot", width=2)
            )
        ))
        
        # Apply base styling
        fig = self.chart_styler.apply_base_layout(fig, title)
        fig.update_layout(height=500)
        
        return fig
    
    def create_drop_off_bar_chart(self, funnel_data: pd.DataFrame) -> go.Figure:
        """Create a bar chart showing drop-off rates between stages."""
        funnel_data = funnel_data.sort_values('stage_order')
        
        # Calculate stage-to-stage drop-off rates
        drop_off_data = []
        for i in range(1, len(funnel_data)):
            prev_stage = funnel_data.iloc[i-1]['stage']
            curr_stage = funnel_data.iloc[i]['stage']
            prev_players = funnel_data.iloc[i-1]['players']
            curr_players = funnel_data.iloc[i]['players']
            
            drop_off_rate = (prev_players - curr_players) / prev_players if prev_players > 0 else 0
            drop_off_players = prev_players - curr_players
            
            drop_off_data.append({
                'transition': f"{prev_stage} → {curr_stage}",
                'drop_off_rate': drop_off_rate,
                'drop_off_players': drop_off_players
            })
        
        drop_off_df = pd.DataFrame(drop_off_data)
        
        # Create bar chart
        fig = go.Figure(data=go.Bar(
            x=drop_off_df['transition'],
            y=drop_off_df['drop_off_rate'],
            marker_color=DashTheme.WARNING_COLOR,
            text=[f"{rate:.1%}" for rate in drop_off_df['drop_off_rate']],
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>' +
                         'Drop-off Rate: %{y:.1%}<br>' +
                         'Players Lost: %{customdata:,}<br>' +
                         '<extra></extra>',
            customdata=drop_off_df['drop_off_players']
        ))
        
        fig = self.chart_styler.apply_base_layout(fig, "Stage-to-Stage Drop-off Rates")
        fig.update_layout(
            xaxis_title="Stage Transition",
            yaxis_title="Drop-off Rate",
            yaxis_tickformat=".0%",
            height=350
        )
        
        # Rotate x-axis labels for better readability
        fig.update_xaxes(tickangle=45)
        
        return fig
    
    def create_level_progression_heatmap(self, level_data: pd.DataFrame) -> go.Figure:
        """Create a heatmap showing player progression through game levels."""
        # Pivot data for heatmap
        heatmap_data = level_data.pivot(
            index='level_group',
            columns='level',
            values='completion_rate'
        )
        
        # Create hover text
        hover_text = []
        for i, level_group in enumerate(heatmap_data.index):
            hover_row = []
            for j, level in enumerate(heatmap_data.columns):
                completion_rate = heatmap_data.iloc[i, j]
                if pd.isna(completion_rate):
                    hover_text_cell = "No data"
                else:
                    hover_text_cell = (
                        f"Level Group: {level_group}<br>"
                        f"Level: {level}<br>"
                        f"Completion Rate: {completion_rate:.1%}"
                    )
                hover_row.append(hover_text_cell)
            hover_text.append(hover_row)
        
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=heatmap_data.columns,
            y=heatmap_data.index,
            hovertemplate='%{text}<extra></extra>',
            text=hover_text,
            colorscale='RdYlGn',
            zmin=0,
            zmax=1,
            colorbar=dict(
                title="Completion Rate",
                tickformat=".0%"
            )
        ))
        
        fig = self.chart_styler.apply_heatmap_styling(fig, "Level Completion Heatmap")
        fig.update_layout(
            xaxis_title="Level",
            yaxis_title="Level Group",
            height=400
        )
        
        return fig
    
    def create_cohort_funnel_comparison(self, cohort_funnel_data: pd.DataFrame) -> go.Figure:
        """Create a comparison of funnel performance across different cohorts."""
        fig = go.Figure()
        
        cohorts = cohort_funnel_data['cohort'].unique()
        colors = self.chart_styler.get_color_palette(len(cohorts))
        
        for i, cohort in enumerate(cohorts):
            cohort_data = cohort_funnel_data[cohort_funnel_data['cohort'] == cohort]
            cohort_data = cohort_data.sort_values('stage_order')
            
            fig.add_trace(go.Scatter(
                x=cohort_data['stage'],
                y=cohort_data['conversion_rate'],
                mode='lines+markers',
                name=f"Cohort {cohort}",
                line=dict(color=colors[i], width=3),
                marker=dict(size=8),
                hovertemplate=f'<b>Cohort {cohort}</b><br>' +
                             'Stage: %{x}<br>' +
                             'Conversion Rate: %{y:.1%}<br>' +
                             '<extra></extra>'
            ))
        
        fig = self.chart_styler.apply_base_layout(fig, "Funnel Performance by Cohort")
        fig.update_layout(
            xaxis_title="Game Stage",
            yaxis_title="Conversion Rate",
            yaxis_tickformat=".0%",
            height=350,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig


def create_dropoff_funnel_component(component_id: str = "dropoff-funnel") -> html.Div:
    """
    Create the complete drop-off funnel component with drill-down functionality.
    
    Args:
        component_id: Base ID for the component
        
    Returns:
        Dash HTML component
    """
    return html.Div([
        # Filters row
        dbc.Row([
            dbc.Col([
                html.Label("Time Period:", className="fw-bold"),
                ComponentFactory.create_date_picker(
                    start_date=(datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
                    end_date=datetime.now().strftime('%Y-%m-%d'),
                    picker_id=f"{component_id}-date-picker"
                )
            ], width=3),
            dbc.Col([
                html.Label("Player Cohort:", className="fw-bold"),
                ComponentFactory.create_filter_dropdown(
                    options=[
                        {"label": "All Cohorts", "value": "all"},
                        {"label": "Week 1", "value": "week1"},
                        {"label": "Week 2", "value": "week2"},
                        {"label": "Week 3", "value": "week3"},
                        {"label": "Week 4", "value": "week4"}
                    ],
                    value="all",
                    dropdown_id=f"{component_id}-cohort-filter",
                    placeholder="Select cohort..."
                )
            ], width=3),
            dbc.Col([
                html.Label("Game Mode:", className="fw-bold"),
                ComponentFactory.create_filter_dropdown(
                    options=[
                        {"label": "All Modes", "value": "all"},
                        {"label": "Tutorial", "value": "tutorial"},
                        {"label": "Campaign", "value": "campaign"},
                        {"label": "Multiplayer", "value": "multiplayer"}
                    ],
                    value="all",
                    dropdown_id=f"{component_id}-mode-filter",
                    placeholder="Select game mode..."
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
                    dropdown_id=f"{component_id}-view-type",
                    placeholder="Select view..."
                )
            ], width=3)
        ], className="mb-3"),
        
        # Main funnel chart
        ComponentFactory.create_loading_wrapper(
            dcc.Graph(
                id=f"{component_id}-main-chart",
                config={
                    'displayModeBar': True,
                    'displaylogo': False,
                    'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d']
                }
            ),
            f"{component_id}-main-loading"
        ),
        
        # Secondary charts row
        dbc.Row([
            dbc.Col([
                ComponentFactory.create_loading_wrapper(
                    dcc.Graph(
                        id=f"{component_id}-dropoff-bars",
                        config={'displayModeBar': False, 'displaylogo': False}
                    ),
                    f"{component_id}-bars-loading"
                )
            ], width=6),
            dbc.Col([
                ComponentFactory.create_loading_wrapper(
                    dcc.Graph(
                        id=f"{component_id}-level-heatmap",
                        config={'displayModeBar': False, 'displaylogo': False}
                    ),
                    f"{component_id}-heatmap-loading"
                )
            ], width=6)
        ], className="mt-3"),
        
        # Drill-down details
        html.Div(id=f"{component_id}-drill-down", className="mt-3"),
        
        # Summary statistics
        html.Div(id=f"{component_id}-summary", className="mt-3")
    ])


def create_funnel_summary_stats(funnel_data: pd.DataFrame) -> dbc.Row:
    """Create summary statistics for the funnel."""
    total_players = funnel_data.iloc[0]['players']
    final_players = funnel_data.iloc[-1]['players']
    overall_conversion = final_players / total_players if total_players > 0 else 0
    
    # Find the stage with highest drop-off
    funnel_data = funnel_data.sort_values('stage_order')
    max_dropoff_idx = 0
    max_dropoff_rate = 0
    
    for i in range(1, len(funnel_data)):
        prev_players = funnel_data.iloc[i-1]['players']
        curr_players = funnel_data.iloc[i]['players']
        dropoff_rate = (prev_players - curr_players) / prev_players if prev_players > 0 else 0
        
        if dropoff_rate > max_dropoff_rate:
            max_dropoff_rate = dropoff_rate
            max_dropoff_idx = i
    
    biggest_dropoff_stage = funnel_data.iloc[max_dropoff_idx]['stage']
    
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
                "Final Conversion",
                f"{overall_conversion:.1%}",
                f"{final_players:,} players"
            )
        ], width=3),
        dbc.Col([
            ComponentFactory.create_metric_card(
                "Biggest Drop-off",
                f"{max_dropoff_rate:.1%}",
                f"At {biggest_dropoff_stage}"
            )
        ], width=3),
        dbc.Col([
            ComponentFactory.create_metric_card(
                "Funnel Stages",
                str(len(funnel_data)),
                "Total stages"
            )
        ], width=3)
    ])


# Sample data generation functions removed - now using real data from ETL pipeline