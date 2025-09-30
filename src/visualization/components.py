"""
Reusable Dash component library for consistent styling and layout.
"""

import dash_bootstrap_components as dbc
from dash import html, dcc
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any, List, Optional
import logging

from .error_handling import (
    handle_visualization_errors, 
    handle_component_errors,
    visualization_cache,
    health_checker
)

logger = logging.getLogger(__name__)


class DashTheme:
    """Centralized theme configuration for consistent styling."""
    
    # Color palette
    PRIMARY_COLOR = "#2E86AB"
    SECONDARY_COLOR = "#A23B72"
    SUCCESS_COLOR = "#F18F01"
    WARNING_COLOR = "#C73E1D"
    INFO_COLOR = "#6C757D"
    
    # Chart colors
    CHART_COLORS = [
        "#2E86AB", "#A23B72", "#F18F01", "#C73E1D", 
        "#6C757D", "#28A745", "#FFC107", "#DC3545"
    ]
    
    # Layout constants
    CARD_MARGIN = {"margin": "10px"}
    CHART_HEIGHT = 400
    HEATMAP_HEIGHT = 500
    
    # Font settings
    FONT_FAMILY = "Arial, sans-serif"
    TITLE_SIZE = 18
    SUBTITLE_SIZE = 14


class ComponentFactory:
    """Factory class for creating consistent Dash components."""
    
    @staticmethod
    @handle_component_errors("Card Component")
    def create_card(title: str, content: Any, card_id: str = None) -> dbc.Card:
        """Create a styled card component with error handling."""
        try:
            card_props = {"style": DashTheme.CARD_MARGIN}
            if card_id is not None:
                card_props["id"] = card_id
            
            result = dbc.Card([
                dbc.CardHeader(html.H5(title, className="mb-0")),
                dbc.CardBody(content)
            ], **card_props)
            
            health_checker.record_success("card_component")
            return result
            
        except Exception as e:
            health_checker.record_error("card_component")
            logger.error(f"Error creating card component: {e}")
            raise
    
    @staticmethod
    def create_filter_dropdown(
        options: List[Dict[str, str]], 
        value: str, 
        dropdown_id: str,
        placeholder: str = "Select..."
    ) -> dcc.Dropdown:
        """Create a styled dropdown filter."""
        return dcc.Dropdown(
            options=options,
            value=value,
            id=dropdown_id,
            placeholder=placeholder,
            style={"marginBottom": "10px"}
        )
    
    @staticmethod
    def create_date_picker(
        start_date: str,
        end_date: str,
        picker_id: str
    ) -> dcc.DatePickerRange:
        """Create a styled date range picker."""
        return dcc.DatePickerRange(
            id=picker_id,
            start_date=start_date,
            end_date=end_date,
            display_format="YYYY-MM-DD",
            style={"marginBottom": "10px"}
        )
    
    @staticmethod
    def create_metric_card(title: str, value: str, delta: str = None) -> dbc.Card:
        """Create a metric display card."""
        content = [
            html.H3(value, className="text-primary mb-0"),
            html.P(title, className="text-muted mb-0")
        ]
        
        if delta:
            delta_color = "text-success" if delta.startswith("+") else "text-danger"
            content.append(html.Small(delta, className=delta_color))
        
        return dbc.Card(
            dbc.CardBody(content, className="text-center"),
            style=DashTheme.CARD_MARGIN
        )
    
    @staticmethod
    def create_loading_wrapper(component: Any, loading_id: str) -> dcc.Loading:
        """Wrap component with loading spinner."""
        return dcc.Loading(
            id=loading_id,
            children=component,
            type="default"
        )


class ChartStyler:
    """Utility class for applying consistent chart styling."""
    
    @staticmethod
    @handle_visualization_errors("base_chart")
    def apply_base_layout(fig: go.Figure, title: str = None) -> go.Figure:
        """Apply base layout styling to a Plotly figure with error handling."""
        try:
            if fig is None:
                raise ValueError("Figure cannot be None")
            
            fig.update_layout(
                font_family=DashTheme.FONT_FAMILY,
                title_font_size=DashTheme.TITLE_SIZE,
                title=title,
                plot_bgcolor="white",
                paper_bgcolor="white",
                height=DashTheme.CHART_HEIGHT,
                margin=dict(l=50, r=50, t=80, b=50)
            )
            
            # Update axes styling
            fig.update_xaxes(
                showgrid=True,
                gridwidth=1,
                gridcolor="lightgray",
                showline=True,
                linewidth=1,
                linecolor="black"
            )
            
            fig.update_yaxes(
                showgrid=True,
                gridwidth=1,
                gridcolor="lightgray",
                showline=True,
                linewidth=1,
                linecolor="black"
            )
            
            health_checker.record_success("chart_styling")
            return fig
            
        except Exception as e:
            health_checker.record_error("chart_styling")
            logger.error(f"Error applying chart styling: {e}")
            raise
    
    @staticmethod
    def apply_heatmap_styling(fig: go.Figure, title: str = None) -> go.Figure:
        """Apply specific styling for heatmap charts."""
        fig.update_layout(
            font_family=DashTheme.FONT_FAMILY,
            title_font_size=DashTheme.TITLE_SIZE,
            title=title,
            height=DashTheme.HEATMAP_HEIGHT,
            margin=dict(l=100, r=50, t=80, b=100)
        )
        return fig
    
    @staticmethod
    def get_color_palette(n_colors: int) -> List[str]:
        """Get a color palette with specified number of colors."""
        if n_colors <= len(DashTheme.CHART_COLORS):
            return DashTheme.CHART_COLORS[:n_colors]
        else:
            # Generate additional colors using Plotly's color scale
            return px.colors.qualitative.Set3[:n_colors]


class LayoutBuilder:
    """Builder class for creating complex dashboard layouts."""
    
    @staticmethod
    def create_filter_row(filters: List[Any]) -> dbc.Row:
        """Create a row of filter components."""
        cols = []
        col_width = 12 // len(filters) if filters else 12
        
        for filter_component in filters:
            cols.append(dbc.Col(filter_component, width=col_width))
        
        return dbc.Row(cols, className="mb-3")
    
    @staticmethod
    def create_chart_grid(charts: List[tuple], rows: int = 2) -> List[dbc.Row]:
        """Create a grid layout for charts."""
        chart_rows = []
        charts_per_row = len(charts) // rows
        
        for i in range(0, len(charts), charts_per_row):
            row_charts = charts[i:i + charts_per_row]
            cols = []
            
            for chart_title, chart_component in row_charts:
                col_width = 12 // len(row_charts)
                card = ComponentFactory.create_card(chart_title, chart_component)
                cols.append(dbc.Col(card, width=col_width))
            
            chart_rows.append(dbc.Row(cols, className="mb-3"))
        
        return chart_rows
    
    @staticmethod
    def create_sidebar_layout(sidebar_content: Any, main_content: Any) -> dbc.Row:
        """Create a sidebar layout with main content area."""
        return dbc.Row([
            dbc.Col(sidebar_content, width=3, className="bg-light p-3"),
            dbc.Col(main_content, width=9, className="p-3")
        ])