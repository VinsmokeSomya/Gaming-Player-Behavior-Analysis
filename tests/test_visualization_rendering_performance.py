"""
Visualization rendering performance tests with various data sizes.
Tests chart creation, styling, and component rendering speed.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
import time
import psutil
import gc
from typing import List, Dict, Any, Tuple
import plotly.graph_objects as go
import plotly.express as px
from unittest.mock import Mock, patch

from src.visualization.components import (
    ComponentFactory, ChartStyler, LayoutBuilder, DashTheme
)
from src.visualization.error_handling import (
    handle_visualization_errors, ErrorFallbackGenerator, 
    CachedVisualizationManager, VisualizationHealthChecker
)


class TestVisualizationRenderingPerformance:
    """Performance tests for visualization rendering with various data sizes."""
    
    @pytest.fixture
    def data_size_configs(self):
        """Define data size configurations for performance testing."""
        return {
            'small': {'points': 100, 'series': 1},
            'medium': {'points': 1000, 'series': 3},
            'large': {'points': 5000, 'series': 5},
            'xlarge': {'points': 10000, 'series': 10}
        }
    
    @pytest.fixture
    def heatmap_size_configs(self):
        """Define heatmap size configurations for performance testing."""
        return {
            'small': {'rows': 10, 'cols': 30},      # 10 cohorts, 30 days
            'medium': {'rows': 30, 'cols': 90},     # 30 cohorts, 90 days
            'large': {'rows': 50, 'cols': 365},     # 50 cohorts, 1 year
            'xlarge': {'rows': 100, 'cols': 365}    # 100 cohorts, 1 year
        }
    
    def generate_time_series_data(self, num_points: int, num_series: int) -> Dict[str, Any]:
        """Generate synthetic time series data for testing."""
        base_date = datetime(2024, 1, 1)
        dates = [base_date + timedelta(days=i) for i in range(num_points)]
        
        data = {'dates': dates}
        for i in range(num_series):
            # Generate realistic retention-like data with trend and noise
            trend = np.linspace(0.8, 0.4, num_points)  # Declining retention
            noise = np.random.normal(0, 0.05, num_points)
            values = np.clip(trend + noise, 0, 1)
            data[f'series_{i}'] = values
        
        return data
    
    def generate_heatmap_data(self, rows: int, cols: int) -> Tuple[np.ndarray, List[str], List[str]]:
        """Generate synthetic heatmap data for testing."""
        # Generate realistic retention heatmap data
        z_data = np.zeros((rows, cols))
        
        for i in range(rows):
            for j in range(cols):
                # Simulate retention decay over time
                base_retention = np.random.uniform(0.7, 0.9)  # Initial retention
                decay_factor = np.exp(-j / 30.0)  # Exponential decay
                noise = np.random.normal(0, 0.05)
                retention = base_retention * decay_factor + noise
                z_data[i, j] = max(0, min(retention, 1))
        
        x_labels = [f"Day {i+1}" for i in range(cols)]
        y_labels = [f"Cohort {i+1}" for i in range(rows)]
        
        return z_data, x_labels, y_labels
    
    @pytest.mark.performance
    def test_line_chart_rendering_performance(self, data_size_configs):
        """Test line chart rendering performance with various data sizes."""
        results = {}
        
        for config_name, config in data_size_configs.items():
            print(f"\nTesting line chart rendering: {config_name} ({config['points']} points, {config['series']} series)")
            
            # Generate test data
            data = self.generate_time_series_data(config['points'], config['series'])
            
            # Measure rendering performance
            start_time = time.time()
            memory_before = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            # Create line chart
            fig = go.Figure()
            for i in range(config['series']):
                fig.add_scatter(
                    x=data['dates'],
                    y=data[f'series_{i}'],
                    mode='lines+markers',
                    name=f'Series {i+1}',
                    line=dict(color=DashTheme.CHART_COLORS[i % len(DashTheme.CHART_COLORS)])
                )
            
            # Apply styling
            styled_fig = ChartStyler.apply_base_layout(fig, f"Performance Test - {config_name}")
            
            end_time = time.time()
            memory_after = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            duration = end_time - start_time
            memory_used = memory_after - memory_before
            points_per_second = (config['points'] * config['series']) / duration
            
            results[config_name] = {
                'duration': duration,
                'memory_mb': memory_used,
                'points_per_second': points_per_second
            }
            
            # Performance assertions (requirement: < 5s for visualization rendering)
            if config_name == 'small':
                assert duration < 1.0, f"Small chart took {duration:.3f}s, expected < 1.0s"
            elif config_name == 'medium':
                assert duration < 2.0, f"Medium chart took {duration:.3f}s, expected < 2.0s"
            elif config_name == 'large':
                assert duration < 5.0, f"Large chart took {duration:.3f}s, expected < 5.0s"
            elif config_name == 'xlarge':
                assert duration < 10.0, f"XLarge chart took {duration:.3f}s, expected < 10.0s"
            
            assert isinstance(styled_fig, go.Figure)
            assert len(styled_fig.data) == config['series']
            
            print(f"  Duration: {duration:.3f}s")
            print(f"  Memory: {memory_used:.1f}MB")
            print(f"  Points/second: {points_per_second:.0f}")
            
            # Clean up
            del data, fig, styled_fig
            gc.collect()
        
        # Test performance scaling
        if 'small' in results and 'large' in results:
            small_pps = results['small']['points_per_second']
            large_pps = results['large']['points_per_second']
            
            # Performance should not degrade too much with larger datasets
            performance_ratio = large_pps / small_pps
            assert performance_ratio > 0.1, f"Performance degraded too much: {performance_ratio:.3f}"
    
    @pytest.mark.performance
    def test_heatmap_rendering_performance(self, heatmap_size_configs):
        """Test heatmap rendering performance with various matrix sizes."""
        results = {}
        
        for config_name, config in heatmap_size_configs.items():
            print(f"\nTesting heatmap rendering: {config_name} ({config['rows']}x{config['cols']} matrix)")
            
            # Generate test data
            z_data, x_labels, y_labels = self.generate_heatmap_data(config['rows'], config['cols'])
            
            # Measure rendering performance
            start_time = time.time()
            memory_before = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            # Create heatmap
            fig = go.Figure(data=go.Heatmap(
                z=z_data,
                x=x_labels,
                y=y_labels,
                colorscale='RdYlBu_r',
                hoverongaps=False,
                hovertemplate='Cohort: %{y}<br>Day: %{x}<br>Retention: %{z:.2%}<extra></extra>'
            ))
            
            # Apply styling
            styled_fig = ChartStyler.apply_heatmap_styling(
                fig, 
                f"Retention Heatmap - {config_name}"
            )
            
            end_time = time.time()
            memory_after = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            duration = end_time - start_time
            memory_used = memory_after - memory_before
            cells_per_second = (config['rows'] * config['cols']) / duration
            
            results[config_name] = {
                'duration': duration,
                'memory_mb': memory_used,
                'cells_per_second': cells_per_second,
                'total_cells': config['rows'] * config['cols']
            }
            
            # Performance assertions
            if config_name == 'small':
                assert duration < 1.0, f"Small heatmap took {duration:.3f}s, expected < 1.0s"
            elif config_name == 'medium':
                assert duration < 3.0, f"Medium heatmap took {duration:.3f}s, expected < 3.0s"
            elif config_name == 'large':
                assert duration < 5.0, f"Large heatmap took {duration:.3f}s, expected < 5.0s"
            elif config_name == 'xlarge':
                assert duration < 10.0, f"XLarge heatmap took {duration:.3f}s, expected < 10.0s"
            
            assert isinstance(styled_fig, go.Figure)
            
            print(f"  Duration: {duration:.3f}s")
            print(f"  Memory: {memory_used:.1f}MB")
            print(f"  Cells/second: {cells_per_second:.0f}")
            print(f"  Total cells: {results[config_name]['total_cells']:,}")
            
            # Clean up
            del z_data, x_labels, y_labels, fig, styled_fig
            gc.collect()
    
    @pytest.mark.performance
    def test_scatter_plot_rendering_performance(self):
        """Test scatter plot rendering performance with large datasets."""
        data_sizes = [100, 500, 1000, 5000, 10000]
        results = {}
        
        for size in data_sizes:
            print(f"\nTesting scatter plot rendering: {size:,} points")
            
            # Generate scatter plot data
            x_data = np.random.uniform(0, 100, size)
            y_data = np.random.uniform(0, 1, size)  # Churn probabilities
            colors = np.random.choice(['High Risk', 'Medium Risk', 'Low Risk'], size)
            
            # Measure rendering performance
            start_time = time.time()
            
            # Create scatter plot
            fig = go.Figure()
            
            # Group by color for better performance
            for risk_level in ['High Risk', 'Medium Risk', 'Low Risk']:
                mask = colors == risk_level
                if np.any(mask):
                    fig.add_scatter(
                        x=x_data[mask],
                        y=y_data[mask],
                        mode='markers',
                        name=risk_level,
                        marker=dict(
                            size=6,
                            opacity=0.7
                        )
                    )
            
            # Apply styling
            styled_fig = ChartStyler.apply_base_layout(fig, f"Churn Risk Scatter ({size:,} points)")
            
            end_time = time.time()
            duration = end_time - start_time
            points_per_second = size / duration
            
            results[size] = {
                'duration': duration,
                'points_per_second': points_per_second
            }
            
            # Performance assertions
            assert duration < 5.0, f"Scatter plot with {size} points took {duration:.3f}s, expected < 5.0s"
            assert points_per_second > 1000, f"Scatter plot performance {points_per_second:.0f} points/s < 1000"
            
            print(f"  Duration: {duration:.3f}s")
            print(f"  Points/second: {points_per_second:.0f}")
            
            # Clean up
            del x_data, y_data, colors, fig, styled_fig
            gc.collect()
    
    @pytest.mark.performance
    def test_multiple_chart_creation_performance(self):
        """Test performance when creating multiple charts simultaneously."""
        chart_configs = [
            {'type': 'line', 'points': 500, 'series': 2},
            {'type': 'bar', 'categories': 10, 'series': 3},
            {'type': 'scatter', 'points': 1000},
            {'type': 'heatmap', 'rows': 20, 'cols': 50},
            {'type': 'histogram', 'points': 2000}
        ]
        
        print(f"\nTesting multiple chart creation: {len(chart_configs)} charts")
        
        start_time = time.time()
        memory_before = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        charts = []
        
        for i, config in enumerate(chart_configs):
            if config['type'] == 'line':
                # Line chart
                data = self.generate_time_series_data(config['points'], config['series'])
                fig = go.Figure()
                for j in range(config['series']):
                    fig.add_scatter(
                        x=data['dates'],
                        y=data[f'series_{j}'],
                        mode='lines',
                        name=f'Series {j+1}'
                    )
            
            elif config['type'] == 'bar':
                # Bar chart
                categories = [f'Category {j+1}' for j in range(config['categories'])]
                fig = go.Figure()
                for j in range(config['series']):
                    values = np.random.uniform(10, 100, config['categories'])
                    fig.add_bar(x=categories, y=values, name=f'Series {j+1}')
            
            elif config['type'] == 'scatter':
                # Scatter plot
                x_data = np.random.uniform(0, 100, config['points'])
                y_data = np.random.uniform(0, 1, config['points'])
                fig = go.Figure()
                fig.add_scatter(x=x_data, y=y_data, mode='markers', name='Data')
            
            elif config['type'] == 'heatmap':
                # Heatmap
                z_data, x_labels, y_labels = self.generate_heatmap_data(config['rows'], config['cols'])
                fig = go.Figure(data=go.Heatmap(z=z_data, x=x_labels, y=y_labels))
            
            elif config['type'] == 'histogram':
                # Histogram
                data = np.random.normal(50, 15, config['points'])
                fig = go.Figure()
                fig.add_histogram(x=data, nbinsx=30, name='Distribution')
            
            # Apply styling
            styled_fig = ChartStyler.apply_base_layout(fig, f"Chart {i+1} ({config['type']})")
            
            # Create card component
            card = ComponentFactory.create_card(f"Chart {i+1}", styled_fig, f"chart-{i}")
            charts.append(card)
        
        end_time = time.time()
        memory_after = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        duration = end_time - start_time
        memory_used = memory_after - memory_before
        charts_per_second = len(chart_configs) / duration
        
        # Performance assertions
        assert duration < 30.0, f"Multiple chart creation took {duration:.2f}s, expected < 30.0s"
        assert charts_per_second > 0.2, f"Chart creation rate {charts_per_second:.2f}/s too slow"
        assert len(charts) == len(chart_configs)
        
        print(f"  Duration: {duration:.2f}s")
        print(f"  Memory: {memory_used:.1f}MB")
        print(f"  Charts/second: {charts_per_second:.2f}")
    
    @pytest.mark.performance
    def test_component_factory_performance(self):
        """Test ComponentFactory performance with various component types."""
        num_components = 100
        component_types = ['card', 'metric_card', 'dropdown', 'date_picker']
        
        print(f"\nTesting ComponentFactory performance: {num_components} components")
        
        start_time = time.time()
        
        components = []
        for i in range(num_components):
            component_type = component_types[i % len(component_types)]
            
            if component_type == 'card':
                content = f"Card content {i}"
                component = ComponentFactory.create_card(f"Card {i}", content, f"card-{i}")
            
            elif component_type == 'metric_card':
                component = ComponentFactory.create_metric_card(
                    f"Metric {i}", 
                    f"{i * 10}", 
                    f"+{i}%"
                )
            
            elif component_type == 'dropdown':
                options = [{'label': f'Option {j}', 'value': f'opt_{j}'} for j in range(5)]
                component = ComponentFactory.create_filter_dropdown(
                    options, 
                    'opt_0', 
                    f'dropdown-{i}',
                    f'Select option {i}'
                )
            
            elif component_type == 'date_picker':
                component = ComponentFactory.create_date_picker(
                    '2024-01-01',
                    '2024-01-31',
                    f'date-picker-{i}'
                )
            
            components.append(component)
        
        end_time = time.time()
        duration = end_time - start_time
        components_per_second = num_components / duration
        
        # Performance assertions
        assert duration < 5.0, f"Component creation took {duration:.2f}s, expected < 5.0s"
        assert components_per_second > 50, f"Component creation rate {components_per_second:.0f}/s too slow"
        assert len(components) == num_components
        
        print(f"  Duration: {duration:.3f}s")
        print(f"  Components/second: {components_per_second:.0f}")
    
    @pytest.mark.performance
    def test_layout_builder_performance(self):
        """Test LayoutBuilder performance with complex layouts."""
        print("\nTesting LayoutBuilder performance")
        
        # Create test components
        num_charts = 12
        charts = []
        
        for i in range(num_charts):
            # Create simple chart
            fig = go.Figure()
            fig.add_scatter(
                x=list(range(100)),
                y=np.random.uniform(0, 1, 100),
                mode='lines',
                name=f'Chart {i+1}'
            )
            styled_fig = ChartStyler.apply_base_layout(fig, f"Chart {i+1}")
            charts.append((f"Chart {i+1}", styled_fig))
        
        # Measure layout creation performance
        start_time = time.time()
        
        # Create filter row
        filters = [
            ComponentFactory.create_filter_dropdown(
                [{'label': 'All', 'value': 'all'}], 
                'all', 
                'filter-1'
            ),
            ComponentFactory.create_date_picker('2024-01-01', '2024-01-31', 'date-1')
        ]
        filter_row = LayoutBuilder.create_filter_row(filters)
        
        # Create chart grid
        chart_rows = LayoutBuilder.create_chart_grid(charts, rows=3)
        
        # Create sidebar layout
        sidebar_content = ComponentFactory.create_card("Sidebar", "Sidebar content")
        main_content = ComponentFactory.create_card("Main", chart_rows)
        sidebar_layout = LayoutBuilder.create_sidebar_layout(sidebar_content, main_content)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Performance assertions
        assert duration < 5.0, f"Layout creation took {duration:.3f}s, expected < 5.0s"
        assert filter_row is not None
        assert len(chart_rows) == 3  # 12 charts in 3 rows
        assert sidebar_layout is not None
        
        print(f"  Duration: {duration:.3f}s")
        print(f"  Charts processed: {num_charts}")
        print(f"  Layout rows created: {len(chart_rows)}")
    
    @pytest.mark.performance
    def test_error_handling_performance(self):
        """Test error handling and fallback performance."""
        print("\nTesting error handling performance")
        
        # Test error fallback generation
        num_errors = 50
        start_time = time.time()
        
        error_charts = []
        for i in range(num_errors):
            error_chart = ErrorFallbackGenerator.create_error_chart(
                f"Test error {i}", 
                f"chart_{i}"
            )
            error_charts.append(error_chart)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Performance assertions
        assert duration < 2.0, f"Error chart generation took {duration:.3f}s, expected < 2.0s"
        assert len(error_charts) == num_errors
        assert all(isinstance(chart, go.Figure) for chart in error_charts)
        
        print(f"  Error chart generation: {duration:.3f}s for {num_errors} charts")
        
        # Test cached visualization manager performance
        cache_manager = CachedVisualizationManager(cache_duration_minutes=1)
        
        start_time = time.time()
        
        # Cache and retrieve operations
        for i in range(100):
            key = f"test_key_{i}"
            data = {'chart_data': list(range(100)), 'metadata': f'test_{i}'}
            
            # Cache operation
            cache_manager.cache_result(key, data)
            
            # Retrieve operation
            cached_data = cache_manager.get_cached_result(key)
            assert cached_data == data
        
        end_time = time.time()
        cache_duration = end_time - start_time
        
        assert cache_duration < 1.0, f"Cache operations took {cache_duration:.3f}s, expected < 1.0s"
        
        print(f"  Cache operations: {cache_duration:.3f}s for 100 operations")
    
    @pytest.mark.performance
    def test_health_checker_performance(self):
        """Test visualization health checker performance."""
        print("\nTesting health checker performance")
        
        health_checker = VisualizationHealthChecker()
        
        # Test health tracking performance
        num_operations = 1000
        components = [f'component_{i}' for i in range(10)]
        
        start_time = time.time()
        
        for i in range(num_operations):
            component = components[i % len(components)]
            
            if i % 10 == 0:  # 10% error rate
                health_checker.record_error(component)
            else:
                health_checker.record_success(component)
        
        # Generate health report
        report = health_checker.get_health_report()
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Performance assertions
        assert duration < 1.0, f"Health tracking took {duration:.3f}s, expected < 1.0s"
        assert 'error_counts' in report
        assert 'last_success' in report
        
        operations_per_second = num_operations / duration
        print(f"  Duration: {duration:.3f}s")
        print(f"  Operations/second: {operations_per_second:.0f}")
        print(f"  Components tracked: {len(components)}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "-m", "performance"])