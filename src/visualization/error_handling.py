"""
Error handling utilities for visualization components.
"""

import logging
import functools
from typing import Any, Dict, Optional, Callable
import plotly.graph_objects as go
import plotly.express as px
from dash import html, dcc
import dash_bootstrap_components as dbc
import traceback

logger = logging.getLogger(__name__)


class VisualizationError(Exception):
    """Custom exception for visualization errors."""
    pass


class ErrorFallbackGenerator:
    """Generates fallback content when visualizations fail."""
    
    @staticmethod
    def create_error_chart(error_message: str, chart_type: str = "generic") -> go.Figure:
        """Create a fallback chart displaying error information."""
        fig = go.Figure()
        
        # Add error message as annotation
        fig.add_annotation(
            text=f"⚠️ Visualization Error<br><br>{error_message}<br><br>Please try refreshing or contact support",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            xanchor='center', yanchor='middle',
            showarrow=False,
            font=dict(size=16, color="red"),
            bgcolor="rgba(255, 0, 0, 0.1)",
            bordercolor="red",
            borderwidth=2
        )
        
        # Basic layout
        fig.update_layout(
            title=f"Error in {chart_type.title()} Chart",
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            plot_bgcolor="white",
            paper_bgcolor="white",
            height=400
        )
        
        return fig
    
    @staticmethod
    def create_error_component(error_message: str, component_name: str = "Component") -> html.Div:
        """Create a fallback Dash component for errors."""
        return dbc.Alert([
            html.H5(f"⚠️ {component_name} Error", className="alert-heading"),
            html.P(error_message),
            html.Hr(),
            html.P("This error has been logged. Please try refreshing the page.", className="mb-0")
        ], color="danger", dismissable=True)
    
    @staticmethod
    def create_loading_placeholder(component_name: str = "Component") -> html.Div:
        """Create a loading placeholder component."""
        return dbc.Card([
            dbc.CardBody([
                dcc.Loading(
                    children=[html.Div(f"Loading {component_name}...")],
                    type="default"
                )
            ])
        ])
    
    @staticmethod
    def create_no_data_chart(message: str = "No data available") -> go.Figure:
        """Create a chart for when no data is available."""
        fig = go.Figure()
        
        fig.add_annotation(
            text=f"📊 {message}<br><br>Try adjusting your filters or date range",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            xanchor='center', yanchor='middle',
            showarrow=False,
            font=dict(size=16, color="gray"),
            bgcolor="rgba(128, 128, 128, 0.1)",
            bordercolor="gray",
            borderwidth=1
        )
        
        fig.update_layout(
            title="No Data Available",
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            plot_bgcolor="white",
            paper_bgcolor="white",
            height=400
        )
        
        return fig


def handle_visualization_errors(chart_type: str = "chart", fallback_data: Optional[Dict] = None):
    """Decorator for handling visualization errors gracefully."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
                
                # Check if result is empty/None and provide fallback
                if result is None:
                    logger.warning(f"{func.__name__} returned None, using fallback")
                    return ErrorFallbackGenerator.create_no_data_chart(
                        f"No data available for {chart_type}"
                    )
                
                # For Plotly figures, check if they have data
                if isinstance(result, go.Figure) and not result.data:
                    logger.warning(f"{func.__name__} returned empty figure, using fallback")
                    return ErrorFallbackGenerator.create_no_data_chart(
                        f"No data available for {chart_type}"
                    )
                
                return result
                
            except Exception as e:
                error_msg = str(e)
                logger.error(f"Error in {func.__name__}: {error_msg}")
                logger.debug(f"Full traceback: {traceback.format_exc()}")
                
                # Return appropriate fallback based on expected return type
                if 'figure' in func.__name__.lower() or chart_type:
                    return ErrorFallbackGenerator.create_error_chart(error_msg, chart_type)
                else:
                    return ErrorFallbackGenerator.create_error_component(error_msg, func.__name__)
        
        return wrapper
    return decorator


def handle_component_errors(component_name: str = "Component"):
    """Decorator for handling Dash component errors."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_msg = str(e)
                logger.error(f"Error in {func.__name__}: {error_msg}")
                logger.debug(f"Full traceback: {traceback.format_exc()}")
                
                return ErrorFallbackGenerator.create_error_component(error_msg, component_name)
        
        return wrapper
    return decorator


class CachedVisualizationManager:
    """Manages cached visualizations for fallback scenarios."""
    
    def __init__(self, cache_duration_minutes: int = 30):
        self.cache = {}
        self.cache_duration = cache_duration_minutes
    
    def get_cached_result(self, cache_key: str) -> Optional[Any]:
        """Get cached visualization result if available and not expired."""
        if cache_key in self.cache:
            cached_item = self.cache[cache_key]
            if self._is_cache_valid(cached_item['timestamp']):
                logger.info(f"Using cached result for {cache_key}")
                return cached_item['data']
        return None
    
    def cache_result(self, cache_key: str, data: Any) -> None:
        """Cache visualization result."""
        from datetime import datetime
        self.cache[cache_key] = {
            'data': data,
            'timestamp': datetime.now()
        }
        logger.debug(f"Cached result for {cache_key}")
    
    def _is_cache_valid(self, timestamp) -> bool:
        """Check if cached item is still valid."""
        from datetime import datetime, timedelta
        return datetime.now() - timestamp < timedelta(minutes=self.cache_duration)
    
    def clear_cache(self) -> None:
        """Clear all cached results."""
        self.cache.clear()
        logger.info("Visualization cache cleared")


def with_caching(cache_key_func: Callable = None, cache_manager: CachedVisualizationManager = None):
    """Decorator to add caching to visualization functions."""
    if cache_manager is None:
        cache_manager = CachedVisualizationManager()
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if cache_key_func:
                cache_key = cache_key_func(*args, **kwargs)
            else:
                cache_key = f"{func.__name__}_{hash(str(args) + str(kwargs))}"
            
            # Try to get cached result
            cached_result = cache_manager.get_cached_result(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Generate new result
            try:
                result = func(*args, **kwargs)
                cache_manager.cache_result(cache_key, result)
                return result
            except Exception as e:
                # Try to return stale cached data as fallback
                if cache_key in cache_manager.cache:
                    logger.warning(f"Using stale cached data for {cache_key} due to error: {e}")
                    return cache_manager.cache[cache_key]['data']
                raise
        
        return wrapper
    return decorator


class VisualizationHealthChecker:
    """Monitors visualization component health."""
    
    def __init__(self):
        self.error_counts = {}
        self.last_success = {}
    
    def record_error(self, component_name: str) -> None:
        """Record an error for a component."""
        self.error_counts[component_name] = self.error_counts.get(component_name, 0) + 1
        logger.warning(f"Error count for {component_name}: {self.error_counts[component_name]}")
    
    def record_success(self, component_name: str) -> None:
        """Record a successful operation for a component."""
        from datetime import datetime
        self.last_success[component_name] = datetime.now()
        # Reset error count on success
        if component_name in self.error_counts:
            self.error_counts[component_name] = 0
    
    def is_component_healthy(self, component_name: str, max_errors: int = 5) -> bool:
        """Check if a component is healthy based on recent errors."""
        error_count = self.error_counts.get(component_name, 0)
        return error_count < max_errors
    
    def get_health_report(self) -> Dict[str, Any]:
        """Get health report for all components."""
        return {
            'error_counts': self.error_counts.copy(),
            'last_success': {k: v.isoformat() for k, v in self.last_success.items()},
            'unhealthy_components': [
                name for name, count in self.error_counts.items() if count >= 5
            ]
        }


# Global instances
visualization_cache = CachedVisualizationManager()
health_checker = VisualizationHealthChecker()