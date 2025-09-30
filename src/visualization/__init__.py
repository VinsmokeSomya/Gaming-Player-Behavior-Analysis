# Visualization components for charts and reports

from .components import (
    DashTheme,
    ComponentFactory,
    ChartStyler,
    LayoutBuilder
)

from .cohort_heatmap import (
    CohortHeatmapGenerator,
    create_cohort_heatmap_component,
    generate_sample_cohort_data
)

from .engagement_timeline import (
    EngagementTimelineGenerator,
    create_engagement_timeline_component,
    generate_sample_engagement_data
)

from .churn_histogram import (
    ChurnHistogramGenerator,
    create_churn_histogram_component,
    create_churn_stats_summary,
    generate_sample_churn_data
)

from .dropoff_funnel import (
    DropoffFunnelGenerator,
    create_dropoff_funnel_component,
    create_funnel_summary_stats,
    generate_sample_funnel_data,
    generate_sample_level_data,
    generate_sample_cohort_funnel_data
)

__all__ = [
    # Core components
    'DashTheme',
    'ComponentFactory', 
    'ChartStyler',
    'LayoutBuilder',
    
    # Cohort heatmap
    'CohortHeatmapGenerator',
    'create_cohort_heatmap_component',
    'generate_sample_cohort_data',
    
    # Engagement timeline
    'EngagementTimelineGenerator',
    'create_engagement_timeline_component',
    'generate_sample_engagement_data',
    
    # Churn histogram
    'ChurnHistogramGenerator',
    'create_churn_histogram_component',
    'create_churn_stats_summary',
    'generate_sample_churn_data',
    
    # Drop-off funnel
    'DropoffFunnelGenerator',
    'create_dropoff_funnel_component',
    'create_funnel_summary_stats',
    'generate_sample_funnel_data',
    'generate_sample_level_data',
    'generate_sample_cohort_funnel_data'
]