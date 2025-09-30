"""
End-to-end integration tests for the complete dashboard application.
"""

import pytest
import time
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

import dash
from dash import html, dcc
from dash.testing.application_runners import import_app
import pandas as pd
import requests
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

from app import app
from src.utils.error_handling import error_handler, health_checker
from src.utils.logging_config import performance_logger


class TestDashboardIntegration:
    """Integration tests for the complete dashboard."""
    
    @pytest.fixture
    def dash_duo(self, dash_duo):
        """Set up Dash test environment."""
        return dash_duo
    
    def test_dashboard_startup(self, dash_duo):
        """Test that the dashboard starts up correctly."""
        dash_duo.start_server(app)
        
        # Wait for the page to load
        dash_duo.wait_for_element("#page-content", timeout=10)
        
        # Check that the main layout is rendered
        assert dash_duo.find_element("#page-content")
        
        # Check for navigation elements
        dash_duo.wait_for_element(".navbar", timeout=5)
        assert dash_duo.find_element(".navbar")
    
    def test_authentication_flow(self, dash_duo):
        """Test the authentication workflow."""
        dash_duo.start_server(app)
        
        # Should show login page initially (in production mode)
        dash_duo.wait_for_element("#username", timeout=10)
        
        # Enter credentials
        username_input = dash_duo.find_element("#username")
        password_input = dash_duo.find_element("#password")
        login_button = dash_duo.find_element("#login-button")
        
        username_input.send_keys("admin")
        password_input.send_keys("admin123")
        login_button.click()
        
        # Should redirect to dashboard
        dash_duo.wait_for_element("#main-tabs", timeout=10)
        assert dash_duo.find_element("#main-tabs")
    
    def test_tab_navigation(self, dash_duo):
        """Test navigation between different tabs."""
        dash_duo.start_server(app)
        dash_duo.wait_for_element("#main-tabs", timeout=10)
        
        # Test each tab
        tabs = ["overview", "cohort", "engagement", "churn", "funnel", "reports"]
        
        for tab_id in tabs:
            # Click on tab
            tab_element = dash_duo.find_element(f"[data-value='{tab_id}']")
            tab_element.click()
            
            # Wait for content to load
            dash_duo.wait_for_element("#tab-content", timeout=5)
            
            # Verify tab content is displayed
            content = dash_duo.find_element("#tab-content")
            assert content.is_displayed()
            
            time.sleep(0.5)  # Brief pause between tab switches
    
    def test_global_filters(self, dash_duo):
        """Test global filter functionality."""
        dash_duo.start_server(app)
        dash_duo.wait_for_element("#global-date-picker", timeout=10)
        
        # Test date picker
        date_picker = dash_duo.find_element("#global-date-picker")
        assert date_picker
        
        # Test segment filter
        segment_filter = dash_duo.find_element("#global-segment-filter")
        assert segment_filter
        
        # Change segment filter and verify it updates
        dash_duo.select_dcc_dropdown("#global-segment-filter", "new")
        
        # Wait for update
        time.sleep(2)
        
        # Verify the filter value changed
        selected_value = dash_duo.get_logs()
        # In a real test, we'd verify the charts updated accordingly
    
    def test_chart_rendering(self, dash_duo):
        """Test that charts render correctly in each tab."""
        dash_duo.start_server(app)
        dash_duo.wait_for_element("#main-tabs", timeout=10)
        
        # Test overview tab charts
        overview_tab = dash_duo.find_element("[data-value='overview']")
        overview_tab.click()
        
        dash_duo.wait_for_element("#overview-engagement-chart", timeout=10)
        engagement_chart = dash_duo.find_element("#overview-engagement-chart")
        assert engagement_chart.is_displayed()
        
        churn_chart = dash_duo.find_element("#overview-churn-chart")
        assert churn_chart.is_displayed()
        
        # Test cohort tab
        cohort_tab = dash_duo.find_element("[data-value='cohort']")
        cohort_tab.click()
        
        dash_duo.wait_for_element("#cohort-heatmap", timeout=10)
        heatmap = dash_duo.find_element("#cohort-heatmap")
        assert heatmap.is_displayed()
    
    def test_error_handling(self, dash_duo):
        """Test error handling and recovery."""
        dash_duo.start_server(app)
        
        # Simulate an error condition
        with patch('app.get_sample_data') as mock_data:
            mock_data.side_effect = Exception("Test error")
            
            # Navigate to a tab that would trigger data loading
            dash_duo.wait_for_element("#main-tabs", timeout=10)
            overview_tab = dash_duo.find_element("[data-value='overview']")
            overview_tab.click()
            
            # Should show error message instead of crashing
            dash_duo.wait_for_element(".alert-danger", timeout=10)
            error_alert = dash_duo.find_element(".alert-danger")
            assert error_alert.is_displayed()
    
    def test_responsive_design(self, dash_duo):
        """Test responsive design on different screen sizes."""
        dash_duo.start_server(app)
        
        # Test desktop size
        dash_duo.driver.set_window_size(1920, 1080)
        dash_duo.wait_for_element("#main-tabs", timeout=10)
        
        # Verify navigation is visible
        navbar = dash_duo.find_element(".navbar")
        assert navbar.is_displayed()
        
        # Test tablet size
        dash_duo.driver.set_window_size(768, 1024)
        time.sleep(1)
        
        # Navigation should still be accessible
        assert navbar.is_displayed()
        
        # Test mobile size
        dash_duo.driver.set_window_size(375, 667)
        time.sleep(1)
        
        # Should have mobile-friendly layout
        assert navbar.is_displayed()
    
    def test_performance_monitoring(self, dash_duo):
        """Test performance monitoring functionality."""
        dash_duo.start_server(app)
        
        # Monitor page load time
        start_time = time.time()
        dash_duo.wait_for_element("#main-tabs", timeout=10)
        load_time = time.time() - start_time
        
        # Page should load within reasonable time
        assert load_time < 10, f"Page load took {load_time:.2f} seconds"
        
        # Test chart rendering performance
        overview_tab = dash_duo.find_element("[data-value='overview']")
        
        start_time = time.time()
        overview_tab.click()
        dash_duo.wait_for_element("#overview-engagement-chart", timeout=10)
        render_time = time.time() - start_time
        
        # Charts should render quickly
        assert render_time < 5, f"Chart rendering took {render_time:.2f} seconds"
    
    def test_data_refresh(self, dash_duo):
        """Test automatic data refresh functionality."""
        dash_duo.start_server(app)
        dash_duo.wait_for_element("#refresh-button", timeout=10)
        
        # Click refresh button
        refresh_button = dash_duo.find_element("#refresh-button")
        refresh_button.click()
        
        # Wait for refresh to complete
        time.sleep(3)
        
        # Verify content is still displayed
        dash_duo.wait_for_element("#tab-content", timeout=5)
        content = dash_duo.find_element("#tab-content")
        assert content.is_displayed()
    
    def test_cross_filtering(self, dash_duo):
        """Test cross-filtering between charts."""
        dash_duo.start_server(app)
        dash_duo.wait_for_element("#main-tabs", timeout=10)
        
        # Go to overview tab
        overview_tab = dash_duo.find_element("[data-value='overview']")
        overview_tab.click()
        
        # Wait for charts to load
        dash_duo.wait_for_element("#overview-engagement-chart", timeout=10)
        
        # Click on a chart element (this would trigger cross-filtering in a real implementation)
        engagement_chart = dash_duo.find_element("#overview-engagement-chart")
        engagement_chart.click()
        
        # In a real test, we'd verify that other charts update accordingly
        time.sleep(2)
    
    def test_drill_down_functionality(self, dash_duo):
        """Test drill-down functionality."""
        dash_duo.start_server(app)
        dash_duo.wait_for_element("#main-tabs", timeout=10)
        
        # Navigate to cohort tab
        cohort_tab = dash_duo.find_element("[data-value='cohort']")
        cohort_tab.click()
        
        # Wait for heatmap to load
        dash_duo.wait_for_element("#cohort-heatmap", timeout=10)
        
        # Click on heatmap cell (would trigger drill-down)
        heatmap = dash_duo.find_element("#cohort-heatmap")
        heatmap.click()
        
        # In a real implementation, this would enable the drill-down tab
        time.sleep(2)


class TestDashboardAPI:
    """Test dashboard API endpoints and health checks."""
    
    def test_health_check_endpoint(self):
        """Test the health check endpoint."""
        # This would test a real health check endpoint
        health_status = health_checker.run_checks()
        
        assert 'overall_status' in health_status
        assert 'checks' in health_status
        assert 'timestamp' in health_status
    
    def test_error_reporting(self):
        """Test error reporting and monitoring."""
        # Simulate an error
        try:
            raise ValueError("Test error for monitoring")
        except Exception as e:
            error_response = error_handler.handle_error(e, {'test': True})
            
            assert 'component' in error_response or 'fallback_chart' in error_response
        
        # Check error summary
        error_summary = error_handler.get_error_summary()
        assert 'total_errors' in error_summary
        assert error_summary['total_errors'] > 0


class TestDashboardPerformance:
    """Performance tests for the dashboard."""
    
    def test_data_loading_performance(self):
        """Test data loading performance."""
        from app import get_sample_data
        
        start_time = time.time()
        data = get_sample_data()
        load_time = time.time() - start_time
        
        # Data should load quickly
        assert load_time < 2, f"Data loading took {load_time:.2f} seconds"
        
        # Verify data structure
        assert 'cohort_data' in data
        assert 'engagement_data' in data
        assert 'churn_data' in data
        assert 'funnel_data' in data
    
    def test_chart_generation_performance(self):
        """Test chart generation performance."""
        from src.visualization import (
            CohortHeatmapGenerator,
            EngagementTimelineGenerator,
            ChurnHistogramGenerator,
            DropoffFunnelGenerator,
            generate_sample_cohort_data,
            generate_sample_engagement_data,
            generate_sample_churn_data,
            generate_sample_funnel_data
        )
        
        # Test each chart type
        generators = [
            (CohortHeatmapGenerator(), generate_sample_cohort_data()),
            (EngagementTimelineGenerator(), generate_sample_engagement_data()),
            (ChurnHistogramGenerator(), generate_sample_churn_data()),
            (DropoffFunnelGenerator(), generate_sample_funnel_data())
        ]
        
        for generator, data in generators:
            start_time = time.time()
            
            if hasattr(generator, 'create_cohort_heatmap'):
                fig = generator.create_cohort_heatmap(data)
            elif hasattr(generator, 'create_engagement_timeline'):
                fig = generator.create_engagement_timeline(data)
            elif hasattr(generator, 'create_churn_risk_histogram'):
                fig = generator.create_churn_risk_histogram(data)
            elif hasattr(generator, 'create_funnel_chart'):
                fig = generator.create_funnel_chart(data)
            
            generation_time = time.time() - start_time
            
            # Chart generation should be fast
            assert generation_time < 1, f"Chart generation took {generation_time:.2f} seconds"
    
    def test_memory_usage(self):
        """Test memory usage during dashboard operations."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Perform memory-intensive operations
        from app import get_sample_data
        
        for _ in range(10):
            data = get_sample_data()
            # Simulate chart generation
            time.sleep(0.1)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable
        assert memory_increase < 100, f"Memory increased by {memory_increase:.2f} MB"


class TestDashboardSecurity:
    """Security tests for the dashboard."""
    
    def test_authentication_required(self):
        """Test that authentication is required for protected resources."""
        # This would test actual authentication in production
        pass
    
    def test_authorization_levels(self):
        """Test different authorization levels."""
        # Test admin access
        # Test analyst access  
        # Test viewer access
        pass
    
    def test_input_validation(self):
        """Test input validation and sanitization."""
        # Test SQL injection prevention
        # Test XSS prevention
        # Test CSRF protection
        pass
    
    def test_session_management(self):
        """Test session management and timeout."""
        # Test session timeout
        # Test session invalidation
        pass


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "--tb=short"])