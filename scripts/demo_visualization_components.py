#!/usr/bin/env python3
"""
Demo script for visualization components.
This script demonstrates the interactive visualization components created for player retention analytics.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

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


def demo_cohort_heatmap():
    """Demonstrate cohort heatmap generation."""
    print("=== Cohort Heatmap Demo ===")
    
    # Generate sample data
    cohort_data = generate_sample_cohort_data()
    print(f"Generated cohort data with {len(cohort_data)} records")
    print(f"Date range: {cohort_data['cohort_date'].min()} to {cohort_data['cohort_date'].max()}")
    print(f"Period range: {cohort_data['period'].min()} to {cohort_data['period'].max()} days")
    
    # Create heatmap generator
    generator = CohortHeatmapGenerator()
    
    # Generate heatmap
    fig = generator.create_cohort_heatmap(cohort_data)
    print(f"Created heatmap with {len(fig.data)} traces")
    
    # Generate cohort size chart
    size_fig = generator.create_cohort_size_chart(cohort_data)
    print(f"Created cohort size chart with {len(size_fig.data)} traces")
    
    print("✓ Cohort heatmap demo completed successfully\n")


def demo_engagement_timeline():
    """Demonstrate engagement timeline generation."""
    print("=== Engagement Timeline Demo ===")
    
    # Generate sample data
    engagement_data = generate_sample_engagement_data()
    print(f"Generated engagement data with {len(engagement_data)} records")
    print(f"Date range: {engagement_data['date'].min()} to {engagement_data['date'].max()}")
    print(f"DAU range: {engagement_data['dau'].min():,} to {engagement_data['dau'].max():,}")
    
    # Create timeline generator
    generator = EngagementTimelineGenerator()
    
    # Generate timeline chart
    fig = generator.create_engagement_timeline(engagement_data)
    print(f"Created timeline chart with {len(fig.data)} traces")
    
    # Generate session metrics chart
    session_fig = generator.create_session_metrics_chart(engagement_data)
    print(f"Created session metrics chart with {len(session_fig.data)} traces")
    
    # Generate engagement distribution
    dist_fig = generator.create_engagement_distribution(engagement_data)
    print(f"Created engagement distribution with {len(dist_fig.data)} traces")
    
    print("✓ Engagement timeline demo completed successfully\n")


def demo_churn_histogram():
    """Demonstrate churn histogram generation."""
    print("=== Churn Histogram Demo ===")
    
    # Generate sample data
    churn_data = generate_sample_churn_data()
    print(f"Generated churn data with {len(churn_data)} records")
    print(f"Risk score range: {churn_data['churn_risk_score'].min():.3f} to {churn_data['churn_risk_score'].max():.3f}")
    print(f"Segments: {', '.join(churn_data['segment'].unique())}")
    
    # Create histogram generator
    generator = ChurnHistogramGenerator()
    
    # Generate risk histogram
    fig = generator.create_churn_risk_histogram(churn_data)
    print(f"Created risk histogram with {len(fig.data)} traces")
    
    # Generate risk category breakdown
    pie_fig = generator.create_risk_category_breakdown(churn_data)
    print(f"Created risk category pie chart with {len(pie_fig.data)} traces")
    
    # Generate segment comparison
    box_fig = generator.create_segment_risk_comparison(churn_data)
    print(f"Created segment comparison with {len(box_fig.data)} traces")
    
    # Generate risk vs engagement scatter
    scatter_fig = generator.create_risk_vs_engagement_scatter(churn_data)
    print(f"Created risk vs engagement scatter with {len(scatter_fig.data)} traces")
    
    print("✓ Churn histogram demo completed successfully\n")


def demo_dropoff_funnel():
    """Demonstrate drop-off funnel generation."""
    print("=== Drop-off Funnel Demo ===")
    
    # Generate sample data
    funnel_data = generate_sample_funnel_data()
    print(f"Generated funnel data with {len(funnel_data)} stages")
    print(f"Player range: {funnel_data['players'].min():,} to {funnel_data['players'].max():,}")
    
    # Create funnel generator
    generator = DropoffFunnelGenerator()
    
    # Generate funnel chart
    fig = generator.create_funnel_chart(funnel_data)
    print(f"Created funnel chart with {len(fig.data)} traces")
    
    # Generate drop-off bar chart
    bar_fig = generator.create_drop_off_bar_chart(funnel_data)
    print(f"Created drop-off bar chart with {len(bar_fig.data)} traces")
    
    print("✓ Drop-off funnel demo completed successfully\n")


def demo_data_validation():
    """Demonstrate data validation and formatting."""
    print("=== Data Validation Demo ===")
    
    # Test all sample data generators
    cohort_data = generate_sample_cohort_data()
    engagement_data = generate_sample_engagement_data()
    churn_data = generate_sample_churn_data()
    funnel_data = generate_sample_funnel_data()
    
    # Validate cohort data
    assert all(col in cohort_data.columns for col in ['cohort_date', 'period', 'retention_rate', 'cohort_size'])
    assert cohort_data['retention_rate'].between(0, 1).all()
    print("✓ Cohort data validation passed")
    
    # Validate engagement data
    assert all(col in engagement_data.columns for col in ['date', 'dau', 'wau', 'mau'])
    assert (engagement_data['wau'] >= engagement_data['dau']).all()
    assert (engagement_data['mau'] >= engagement_data['wau']).all()
    print("✓ Engagement data validation passed")
    
    # Validate churn data
    assert all(col in churn_data.columns for col in ['player_id', 'churn_risk_score', 'segment'])
    assert churn_data['churn_risk_score'].between(0, 1).all()
    print("✓ Churn data validation passed")
    
    # Validate funnel data
    assert all(col in funnel_data.columns for col in ['stage', 'players', 'stage_order'])
    sorted_data = funnel_data.sort_values('stage_order')
    players_list = sorted_data['players'].tolist()
    for i in range(1, len(players_list)):
        assert players_list[i] <= players_list[i-1], "Funnel should show decreasing players"
    print("✓ Funnel data validation passed")
    
    print("✓ All data validation tests passed\n")


def main():
    """Run all visualization component demos."""
    print("🎮 Player Retention Analytics - Visualization Components Demo")
    print("=" * 60)
    
    try:
        demo_cohort_heatmap()
        demo_engagement_timeline()
        demo_churn_histogram()
        demo_dropoff_funnel()
        demo_data_validation()
        
        print("🎉 All visualization component demos completed successfully!")
        print("\nComponents created:")
        print("✓ Interactive cohort retention heatmap with hover details and zoom")
        print("✓ Dynamic player engagement timeline charts with date range selection")
        print("✓ Interactive churn risk distribution histograms with segment filtering")
        print("✓ Drop-off funnel visualization with clickable level drill-down")
        print("✓ Reusable Dash component library for consistent styling")
        print("✓ Comprehensive unit tests for all components")
        
        print("\nNext steps:")
        print("- Integrate components into a complete Dash web application")
        print("- Connect to real data sources via analytics query engine")
        print("- Add callback functions for interactive filtering")
        print("- Deploy as a production analytics dashboard")
        
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())