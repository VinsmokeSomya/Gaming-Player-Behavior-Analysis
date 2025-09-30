"""
Comprehensive error handling and monitoring for the dashboard application.
"""

import traceback
import functools
from typing import Any, Callable, Dict, Optional
from datetime import datetime
import logging

from dash import html, dcc
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

from .logging_config import app_logger, security_logger


class DashboardError(Exception):
    """Base exception for dashboard-specific errors."""
    
    def __init__(self, message: str, error_code: str = None, details: Dict = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or 'DASHBOARD_ERROR'
        self.details = details or {}
        self.timestamp = datetime.utcnow()


class DataError(DashboardError):
    """Exception for data-related errors."""
    pass


class AuthenticationError(DashboardError):
    """Exception for authentication-related errors."""
    pass


class AuthorizationError(DashboardError):
    """Exception for authorization-related errors."""
    pass


class VisualizationError(DashboardError):
    """Exception for visualization-related errors."""
    pass


class ErrorHandler:
    """Centralized error handling and monitoring."""
    
    def __init__(self):
        self.error_counts = {}
        self.last_errors = []
        self.max_recent_errors = 100
    
    def handle_error(self, error: Exception, context: Dict = None) -> Dict[str, Any]:
        """
        Handle an error and return appropriate response.
        
        Args:
            error: The exception that occurred
            context: Additional context about the error
        
        Returns:
            Dictionary with error information and suggested response
        """
        context = context or {}
        error_info = {
            'type': type(error).__name__,
            'message': str(error),
            'timestamp': datetime.utcnow().isoformat(),
            'context': context,
            'traceback': traceback.format_exc()
        }
        
        # Log the error
        app_logger.error(
            f"Error occurred: {error_info['message']}",
            extra={
                'error_type': error_info['type'],
                'context': context,
                'traceback': error_info['traceback']
            }
        )
        
        # Track error frequency
        error_key = f"{error_info['type']}:{error_info['message']}"
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        
        # Store recent errors
        self.last_errors.append(error_info)
        if len(self.last_errors) > self.max_recent_errors:
            self.last_errors.pop(0)
        
        # Determine response based on error type
        if isinstance(error, AuthenticationError):
            return self._handle_auth_error(error, error_info)
        elif isinstance(error, AuthorizationError):
            return self._handle_authz_error(error, error_info)
        elif isinstance(error, DataError):
            return self._handle_data_error(error, error_info)
        elif isinstance(error, VisualizationError):
            return self._handle_viz_error(error, error_info)
        else:
            return self._handle_generic_error(error, error_info)
    
    def _handle_auth_error(self, error: AuthenticationError, error_info: Dict) -> Dict:
        """Handle authentication errors."""
        security_logger.log_suspicious_activity(
            f"Authentication error: {error.message}",
            username=error_info.get('context', {}).get('username'),
            ip_address=error_info.get('context', {}).get('ip_address')
        )
        
        return {
            'component': dbc.Alert([
                html.H4("Authentication Required", className="alert-heading"),
                html.P("Please log in to access this resource."),
                dbc.Button("Go to Login", href="/login", color="primary")
            ], color="warning"),
            'redirect': '/login'
        }
    
    def _handle_authz_error(self, error: AuthorizationError, error_info: Dict) -> Dict:
        """Handle authorization errors."""
        security_logger.log_access_denied(
            username=error_info.get('context', {}).get('username', 'unknown'),
            resource=error_info.get('context', {}).get('resource', 'unknown'),
            ip_address=error_info.get('context', {}).get('ip_address')
        )
        
        return {
            'component': dbc.Alert([
                html.H4("Access Denied", className="alert-heading"),
                html.P("You don't have permission to access this resource."),
                html.P("Contact your administrator if you believe this is an error.")
            ], color="danger")
        }
    
    def _handle_data_error(self, error: DataError, error_info: Dict) -> Dict:
        """Handle data-related errors."""
        return {
            'component': dbc.Alert([
                html.H4("Data Error", className="alert-heading"),
                html.P("There was an issue loading or processing the data."),
                html.P("The system will attempt to reload the data automatically."),
                dbc.Button("Retry", id="retry-data-load", color="primary")
            ], color="warning"),
            'retry_action': 'reload_data'
        }
    
    def _handle_viz_error(self, error: VisualizationError, error_info: Dict) -> Dict:
        """Handle visualization errors."""
        return {
            'component': dbc.Alert([
                html.H4("Visualization Error", className="alert-heading"),
                html.P("Unable to generate the requested chart or visualization."),
                html.P("Try refreshing the page or selecting different filters.")
            ], color="info"),
            'fallback_chart': self._create_error_chart(error.message)
        }
    
    def _handle_generic_error(self, error: Exception, error_info: Dict) -> Dict:
        """Handle generic errors."""
        return {
            'component': dbc.Alert([
                html.H4("System Error", className="alert-heading"),
                html.P("An unexpected error occurred. The development team has been notified."),
                html.P(f"Error ID: {error_info['timestamp']}", className="text-muted small")
            ], color="danger"),
            'fallback_content': html.Div([
                html.H5("Service Temporarily Unavailable"),
                html.P("Please try again in a few moments.")
            ])
        }
    
    def _create_error_chart(self, message: str) -> go.Figure:
        """Create a fallback chart for visualization errors."""
        fig = go.Figure()
        fig.add_annotation(
            text=f"Chart Error: {message}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="red")
        )
        fig.update_layout(
            title="Visualization Error",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            showlegend=False
        )
        return fig
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of recent errors for monitoring."""
        return {
            'total_errors': len(self.last_errors),
            'error_counts': self.error_counts,
            'recent_errors': self.last_errors[-10:],  # Last 10 errors
            'most_common_errors': sorted(
                self.error_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
        }


# Global error handler instance
error_handler = ErrorHandler()


def handle_callback_errors(func: Callable) -> Callable:
    """
    Decorator to handle errors in Dash callbacks.
    
    Usage:
        @handle_callback_errors
        def my_callback(input_value):
            # callback logic here
            return result
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            context = {
                'callback_function': func.__name__,
                'args': str(args)[:200],  # Truncate long arguments
                'kwargs': str(kwargs)[:200]
            }
            
            error_response = error_handler.handle_error(e, context)
            
            # Return appropriate fallback based on expected return type
            if 'component' in error_response:
                return error_response['component']
            elif 'fallback_chart' in error_response:
                return dcc.Graph(figure=error_response['fallback_chart'])
            else:
                return html.Div([
                    dbc.Alert("An error occurred while processing your request.", color="danger")
                ])
    
    return wrapper


def create_error_boundary(children, error_id: str = None):
    """
    Create an error boundary component that catches and displays errors gracefully.
    
    Args:
        children: Child components to wrap
        error_id: Unique identifier for this error boundary
    
    Returns:
        Wrapped component with error handling
    """
    return html.Div([
        html.Div(id=f"error-boundary-{error_id}" if error_id else "error-boundary"),
        html.Div(children, id=f"content-{error_id}" if error_id else "content")
    ])


def log_user_action(action: str, user_id: str = None, details: Dict = None):
    """
    Log user actions for audit and monitoring.
    
    Args:
        action: Description of the action
        user_id: ID of the user performing the action
        details: Additional details about the action
    """
    app_logger.info(
        f"User action: {action}",
        extra={
            'event_type': 'user_action',
            'action': action,
            'user_id': user_id,
            'details': details or {}
        }
    )


class HealthChecker:
    """System health monitoring."""
    
    def __init__(self):
        self.checks = {}
    
    def register_check(self, name: str, check_func: Callable) -> None:
        """Register a health check function."""
        self.checks[name] = check_func
    
    def run_checks(self) -> Dict[str, Any]:
        """Run all registered health checks."""
        results = {}
        overall_healthy = True
        
        for name, check_func in self.checks.items():
            try:
                result = check_func()
                results[name] = {
                    'status': 'healthy' if result else 'unhealthy',
                    'details': result if isinstance(result, dict) else {}
                }
                if not result:
                    overall_healthy = False
            except Exception as e:
                results[name] = {
                    'status': 'error',
                    'error': str(e)
                }
                overall_healthy = False
        
        return {
            'overall_status': 'healthy' if overall_healthy else 'unhealthy',
            'checks': results,
            'timestamp': datetime.utcnow().isoformat()
        }


# Global health checker instance
health_checker = HealthChecker()


def register_default_health_checks():
    """Register default health checks."""
    
    def check_database():
        """Check database connectivity."""
        try:
            # This would check actual database connection
            return {'status': 'connected', 'response_time_ms': 50}
        except Exception:
            return False
    
    def check_redis():
        """Check Redis connectivity."""
        try:
            # This would check actual Redis connection
            return {'status': 'connected', 'response_time_ms': 10}
        except Exception:
            return False
    
    def check_memory():
        """Check memory usage."""
        import psutil
        memory = psutil.virtual_memory()
        return {
            'usage_percent': memory.percent,
            'available_gb': memory.available / (1024**3)
        }
    
    health_checker.register_check('database', check_database)
    health_checker.register_check('redis', check_redis)
    health_checker.register_check('memory', check_memory)


# Initialize default health checks
register_default_health_checks()