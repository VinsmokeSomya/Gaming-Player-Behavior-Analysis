"""
Unit tests for visualization components and chart generation.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go

from src.visualization.components import (
    DashTheme, ComponentFactory, ChartStyler, LayoutBuilder
)
from src.visualization.cohort_heatmap import (
    CohortHeatmapGenerator, generate_sample_cohort_data
)
from src.visualization.engagement_timeline import (
    EngagementTimelineGenerator, generate_sample_engagement_data
)
from src.visualization.churn_histogram import (
    ChurnHistogramGenerator, generate_sample_churn_data
)
from src.visualization.dropoff_funnel import (
    DropoffFunnelGenerator, generate_sample_funnel_data,
    generate_sample_level_data, generate_sample_cohort_funnel_data
)


class TestDashTheme:
    """Test the DashTheme configuration."""
    
    def test_color_palette_exists(self):
        """Test that color palette is defined."""
        assert hasattr(DashTheme, 'CHART_COLORS')
        assert len(DashTheme.CHART_COLORS) >= 4
        assert all(color.startswith('#') for color in DashTheme.CHART_COLORS)
    
    def test_layout_constants(self):
        """Test that layout constants are properly defined."""
        assert DashTheme.CHART_HEIGHT > 0
        assert DashTheme.HEATMAP_HEIGHT > 0
        assert DashTheme.TITLE_SIZE > 0
        assert isinstance(DashTheme.CARD_MARGIN, dict)


class TestComponentFactory:
    """Test the ComponentFactory utility functions."""
    
    def test_create_card(self):
        """Test card creation."""
        card = ComponentFactory.create_card("Test Title", "Test Content")
        assert card is not None
        # Card should have proper structure (this is a basic test)
        assert hasattr(card, 'children')
    
    def test_create_filter_dropdown(self):
        """Test dropdown filter creation."""
        options = [{"label": "Option 1", "value": "opt1"}]
        dropdown = ComponentFactory.create_filter_dropdown(
            options, "opt1", "test-dropdown"
        )
        assert dropdown is not None
        assert dropdown.options == options
        assert dropdown.value == "opt1"
        assert dropdown.id == "test-dropdown"
    
    def test_create_metric_card(self):
        """Test metric card creation."""
        card = ComponentFactory.create_metric_card("Test Metric", "100", "+5%")
        assert card is not None
        assert hasattr(card, 'children')


class TestChartStyler:
    """Test the ChartStyler utility functions."""
    
    def test_apply_base_layout(self):
        """Test base layout application."""
        fig = go.Figure()
        styled_fig = ChartStyler.apply_base_layout(fig, "Test Title")
        
        assert styled_fig.layout.title.text == "Test Title"
        assert styled_fig.layout.font.family == DashTheme.FONT_FAMILY
        assert styled_fig.layout.height == DashTheme.CHART_HEIGHT
    
    def test_apply_heatmap_styling(self):
        """Test heatmap styling application."""
        fig = go.Figure()
        styled_fig = ChartStyler.apply_heatmap_styling(fig, "Heatmap Title")
        
        assert styled_fig.layout.title.text == "Heatmap Title"
        assert styled_fig.layout.height == DashTheme.HEATMAP_HEIGHT
    
    def test_get_color_palette(self):
        """Test color palette generation."""
        colors = ChartStyler.get_color_palette(3)
        assert len(colors) == 3
        assert all(isinstance(color, str) for color in colors)


class TestCohortHeatmapGenerator:
    """Test the cohort heatmap generation."""
    
    @pytest.fixture
    def sample_cohort_data(self):
        """Generate sample cohort data for testing."""
        return generate_sample_cohort_data()
    
    @pytest.fixture
    def heatmap_generator(self):
        """Create a heatmap generator instance."""
        return CohortHeatmapGenerator()
    
    def test_create_cohort_heatmap(self, heatmap_generator, sample_cohort_data):
        """Test cohort heatmap creation."""
        fig = heatmap_generator.create_cohort_heatmap(sample_cohort_data)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
        assert fig.data[0].type == 'heatmap'
        assert fig.layout.title.text == "Cohort Retention Heatmap"
    
    def test_create_cohort_size_chart(self, heatmap_generator, sample_cohort_data):
        """Test cohort size chart creation."""
        fig = heatmap_generator.create_cohort_size_chart(sample_cohort_data)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
        assert fig.data[0].type == 'bar'
        assert fig.layout.title.text == "Cohort Sizes"
    
    def test_sample_data_generation(self):
        """Test sample cohort data generation."""
        data = generate_sample_cohort_data()
        
        assert isinstance(data, pd.DataFrame)
        assert not data.empty
        assert all(col in data.columns for col in ['cohort_date', 'period', 'retention_rate', 'cohort_size'])
        assert data['retention_rate'].between(0, 1).all()
        assert (data['cohort_size'] > 0).all()


class TestEngagementTimelineGenerator:
    """Test the engagement timeline generation."""
    
    @pytest.fixture
    def sample_engagement_data(self):
        """Generate sample engagement data for testing."""
        return generate_sample_engagement_data()
    
    @pytest.fixture
    def timeline_generator(self):
        """Create a timeline generator instance."""
        return EngagementTimelineGenerator()
    
    def test_create_engagement_timeline(self, timeline_generator, sample_engagement_data):
        """Test engagement timeline creation."""
        fig = timeline_generator.create_engagement_timeline(sample_engagement_data)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
        assert all(trace.type == 'scatter' for trace in fig.data)
        assert fig.layout.title.text == "Player Engagement Timeline"
    
    def test_create_session_metrics_chart(self, timeline_generator, sample_engagement_data):
        """Test session metrics chart creation."""
        fig = timeline_generator.create_session_metrics_chart(sample_engagement_data)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2  # Two metrics on dual y-axis
        assert fig.layout.title.text == "Session Metrics"
    
    def test_create_engagement_distribution(self, timeline_generator, sample_engagement_data):
        """Test engagement distribution chart creation."""
        fig = timeline_generator.create_engagement_distribution(sample_engagement_data)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
        assert fig.data[0].type == 'histogram'
    
    def test_sample_data_generation(self):
        """Test sample engagement data generation."""
        data = generate_sample_engagement_data()
        
        assert isinstance(data, pd.DataFrame)
        assert not data.empty
        expected_cols = ['date', 'dau', 'wau', 'mau', 'avg_session_duration', 'sessions_per_user']
        assert all(col in data.columns for col in expected_cols)
        assert (data['dau'] > 0).all()
        assert (data['avg_session_duration'] > 0).all()


class TestChurnHistogramGenerator:
    """Test the churn histogram generation."""
    
    @pytest.fixture
    def sample_churn_data(self):
        """Generate sample churn data for testing."""
        return generate_sample_churn_data()
    
    @pytest.fixture
    def histogram_generator(self):
        """Create a histogram generator instance."""
        return ChurnHistogramGenerator()
    
    def test_create_churn_risk_histogram(self, histogram_generator, sample_churn_data):
        """Test churn risk histogram creation."""
        fig = histogram_generator.create_churn_risk_histogram(sample_churn_data)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
        assert all(trace.type == 'histogram' for trace in fig.data)
        assert fig.layout.title.text == "Churn Risk Distribution"
    
    def test_create_risk_category_breakdown(self, histogram_generator, sample_churn_data):
        """Test risk category pie chart creation."""
        fig = histogram_generator.create_risk_category_breakdown(sample_churn_data)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
        assert fig.data[0].type == 'pie'
    
    def test_create_segment_risk_comparison(self, histogram_generator, sample_churn_data):
        """Test segment risk comparison box plot."""
        fig = histogram_generator.create_segment_risk_comparison(sample_churn_data)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
        assert all(trace.type == 'box' for trace in fig.data)
    
    def test_create_risk_vs_engagement_scatter(self, histogram_generator, sample_churn_data):
        """Test risk vs engagement scatter plot."""
        fig = histogram_generator.create_risk_vs_engagement_scatter(sample_churn_data)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
        assert fig.data[0].type == 'scatter'
    
    def test_sample_data_generation(self):
        """Test sample churn data generation."""
        data = generate_sample_churn_data()
        
        assert isinstance(data, pd.DataFrame)
        assert not data.empty
        expected_cols = ['player_id', 'churn_risk_score', 'segment', 'days_since_last_session']
        assert all(col in data.columns for col in expected_cols)
        assert data['churn_risk_score'].between(0, 1).all()
        assert (data['days_since_last_session'] >= 0).all()


class TestDropoffFunnelGenerator:
    """Test the drop-off funnel generation."""
    
    @pytest.fixture
    def sample_funnel_data(self):
        """Generate sample funnel data for testing."""
        return generate_sample_funnel_data()
    
    @pytest.fixture
    def sample_level_data(self):
        """Generate sample level data for testing."""
        return generate_sample_level_data()
    
    @pytest.fixture
    def sample_cohort_funnel_data(self):
        """Generate sample cohort funnel data for testing."""
        return generate_sample_cohort_funnel_data()
    
    @pytest.fixture
    def funnel_generator(self):
        """Create a funnel generator instance."""
        return DropoffFunnelGenerator()
    
    def test_create_funnel_chart(self, funnel_generator, sample_funnel_data):
        """Test funnel chart creation."""
        fig = funnel_generator.create_funnel_chart(sample_funnel_data)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
        assert fig.data[0].type == 'funnel'
        assert fig.layout.title.text == "Player Drop-off Funnel"
    
    def test_create_drop_off_bar_chart(self, funnel_generator, sample_funnel_data):
        """Test drop-off bar chart creation."""
        fig = funnel_generator.create_drop_off_bar_chart(sample_funnel_data)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
        assert fig.data[0].type == 'bar'
    
    def test_create_level_progression_heatmap(self, funnel_generator, sample_level_data):
        """Test level progression heatmap creation."""
        fig = funnel_generator.create_level_progression_heatmap(sample_level_data)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
        assert fig.data[0].type == 'heatmap'
    
    def test_create_cohort_funnel_comparison(self, funnel_generator, sample_cohort_funnel_data):
        """Test cohort funnel comparison chart."""
        fig = funnel_generator.create_cohort_funnel_comparison(sample_cohort_funnel_data)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
        assert all(trace.type == 'scatter' for trace in fig.data)
    
    def test_sample_data_generation(self):
        """Test sample funnel data generation."""
        funnel_data = generate_sample_funnel_data()
        level_data = generate_sample_level_data()
        cohort_data = generate_sample_cohort_funnel_data()
        
        # Test funnel data
        assert isinstance(funnel_data, pd.DataFrame)
        assert not funnel_data.empty
        assert all(col in funnel_data.columns for col in ['stage', 'players', 'stage_order'])
        assert (funnel_data['players'] > 0).all()
        
        # Test level data
        assert isinstance(level_data, pd.DataFrame)
        assert not level_data.empty
        assert all(col in level_data.columns for col in ['level_group', 'level', 'completion_rate'])
        assert level_data['completion_rate'].between(0, 1).all()
        
        # Test cohort data
        assert isinstance(cohort_data, pd.DataFrame)
        assert not cohort_data.empty
        assert all(col in cohort_data.columns for col in ['cohort', 'stage', 'conversion_rate'])
        assert cohort_data['conversion_rate'].between(0, 1).all()


class TestDataFormatting:
    """Test data formatting and validation functions."""
    
    def test_cohort_data_format(self):
        """Test that cohort data has correct format."""
        data = generate_sample_cohort_data()
        
        # Check data types
        assert pd.api.types.is_datetime64_any_dtype(pd.to_datetime(data['cohort_date']))
        assert pd.api.types.is_numeric_dtype(data['period'])
        assert pd.api.types.is_numeric_dtype(data['retention_rate'])
        assert pd.api.types.is_numeric_dtype(data['cohort_size'])
        
        # Check value ranges
        assert data['retention_rate'].between(0, 1).all()
        assert (data['cohort_size'] > 0).all()
        assert (data['period'] >= 0).all()
    
    def test_engagement_data_format(self):
        """Test that engagement data has correct format."""
        data = generate_sample_engagement_data()
        
        # Check data types
        assert pd.api.types.is_datetime64_any_dtype(data['date'])
        assert pd.api.types.is_numeric_dtype(data['dau'])
        assert pd.api.types.is_numeric_dtype(data['avg_session_duration'])
        
        # Check logical relationships
        assert (data['wau'] >= data['dau']).all()  # WAU should be >= DAU
        assert (data['mau'] >= data['wau']).all()  # MAU should be >= WAU
        assert (data['avg_session_duration'] > 0).all()
    
    def test_churn_data_format(self):
        """Test that churn data has correct format."""
        data = generate_sample_churn_data()
        
        # Check data types
        assert pd.api.types.is_string_dtype(data['player_id'])
        assert pd.api.types.is_numeric_dtype(data['churn_risk_score'])
        assert pd.api.types.is_string_dtype(data['segment'])
        
        # Check value ranges
        assert data['churn_risk_score'].between(0, 1).all()
        assert (data['days_since_last_session'] >= 0).all()
        
        # Check segments are valid
        valid_segments = ['new', 'casual', 'core', 'premium']
        assert data['segment'].isin(valid_segments).all()
    
    def test_funnel_data_format(self):
        """Test that funnel data has correct format."""
        data = generate_sample_funnel_data()
        
        # Check data types
        assert pd.api.types.is_string_dtype(data['stage'])
        assert pd.api.types.is_numeric_dtype(data['players'])
        assert pd.api.types.is_numeric_dtype(data['stage_order'])
        
        # Check funnel logic (players should decrease through stages)
        sorted_data = data.sort_values('stage_order')
        players_list = sorted_data['players'].tolist()
        for i in range(1, len(players_list)):
            assert players_list[i] <= players_list[i-1], "Funnel should show decreasing players"


if __name__ == "__main__":
    pytest.main([__file__])