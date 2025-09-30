"""SQL query engine for retention analysis.

This module provides optimized SQL queries for player retention analysis,
including cohort analysis, player segmentation, and drop-off analysis.
"""

import logging
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from sqlalchemy import text
from sqlalchemy.orm import Session

from ..database import db_manager

logger = logging.getLogger(__name__)


@dataclass
class RetentionQueryResult:
    """Result container for retention queries."""
    cohort_date: date
    day_1_retention: float
    day_7_retention: float
    day_30_retention: float
    cohort_size: int
    segment: Optional[str] = None


@dataclass
class DropOffAnalysisResult:
    """Result container for drop-off analysis queries."""
    level: int
    players_reached: int
    players_completed: int
    drop_off_rate: float
    completion_rate: float


@dataclass
class PlayerSegmentResult:
    """Result container for player segmentation queries."""
    segment: str
    player_count: int
    avg_sessions: float
    avg_playtime: float
    avg_retention_day_7: float


class RetentionQueryEngine:
    """SQL query engine for player retention analysis."""
    
    def __init__(self):
        self.db_manager = db_manager
    
    def calculate_cohort_retention(
        self,
        start_date: date,
        end_date: date,
        segment: Optional[str] = None
    ) -> List[RetentionQueryResult]:
        """Calculate retention rates for player cohorts.
        
        Args:
            start_date: Start date for cohort analysis
            end_date: End date for cohort analysis
            segment: Optional player segment filter
            
        Returns:
            List of retention results by cohort date
        """
        query = """
        WITH cohorts AS (
            SELECT 
                DATE(registration_date) as cohort_date,
                player_id,
                registration_date
            FROM player_profiles 
            WHERE DATE(registration_date) BETWEEN :start_date AND :end_date
        ),
        cohort_sizes AS (
            SELECT 
                cohort_date,
                COUNT(*) as cohort_size
            FROM cohorts
            GROUP BY cohort_date
        ),
        day_1_active AS (
            SELECT DISTINCT
                c.cohort_date,
                c.player_id
            FROM cohorts c
            JOIN player_sessions ps ON c.player_id = ps.player_id
            WHERE ps.start_time BETWEEN c.registration_date + INTERVAL '1 day' 
                                   AND c.registration_date + INTERVAL '2 days'
        ),
        day_7_active AS (
            SELECT DISTINCT
                c.cohort_date,
                c.player_id
            FROM cohorts c
            JOIN player_sessions ps ON c.player_id = ps.player_id
            WHERE ps.start_time BETWEEN c.registration_date + INTERVAL '7 days' 
                                   AND c.registration_date + INTERVAL '8 days'
        ),
        day_30_active AS (
            SELECT DISTINCT
                c.cohort_date,
                c.player_id
            FROM cohorts c
            JOIN player_sessions ps ON c.player_id = ps.player_id
            WHERE ps.start_time BETWEEN c.registration_date + INTERVAL '30 days' 
                                   AND c.registration_date + INTERVAL '31 days'
        )
        SELECT 
            cs.cohort_date,
            COALESCE(COUNT(DISTINCT d1.player_id)::DECIMAL / cs.cohort_size, 0) as day_1_retention,
            COALESCE(COUNT(DISTINCT d7.player_id)::DECIMAL / cs.cohort_size, 0) as day_7_retention,
            COALESCE(COUNT(DISTINCT d30.player_id)::DECIMAL / cs.cohort_size, 0) as day_30_retention,
            cs.cohort_size
        FROM cohort_sizes cs
        LEFT JOIN day_1_active d1 ON cs.cohort_date = d1.cohort_date
        LEFT JOIN day_7_active d7 ON cs.cohort_date = d7.cohort_date
        LEFT JOIN day_30_active d30 ON cs.cohort_date = d30.cohort_date
        GROUP BY cs.cohort_date, cs.cohort_size
        ORDER BY cs.cohort_date
        """
        
        params = {
            'start_date': start_date,
            'end_date': end_date
        }
        
        try:
            with self.db_manager.get_session() as session:
                result = session.execute(text(query), params)
                rows = result.fetchall()
                
                return [
                    RetentionQueryResult(
                        cohort_date=row.cohort_date,
                        day_1_retention=float(row.day_1_retention),
                        day_7_retention=float(row.day_7_retention),
                        day_30_retention=float(row.day_30_retention),
                        cohort_size=row.cohort_size,
                        segment=segment
                    )
                    for row in rows
                ]
        except Exception as e:
            logger.error(f"Error calculating cohort retention: {e}")
            raise
    
    def analyze_drop_off_by_level(
        self,
        start_date: date,
        end_date: date,
        max_level: int = 50
    ) -> List[DropOffAnalysisResult]:
        """Analyze player drop-off rates by game level.
        
        Args:
            start_date: Start date for analysis
            end_date: End date for analysis
            max_level: Maximum level to analyze
            
        Returns:
            List of drop-off analysis results by level
        """
        query = """
        WITH level_progression AS (
            SELECT 
                ps.level_reached,
                COUNT(DISTINCT ps.player_id) as players_reached
            FROM player_sessions ps
            JOIN player_profiles pp ON ps.player_id = pp.player_id
            WHERE ps.start_time BETWEEN :start_date AND :end_date
                AND ps.level_reached IS NOT NULL
                AND ps.level_reached <= :max_level
            GROUP BY ps.level_reached
        ),
        level_completion AS (
            SELECT 
                ps.level_reached,
                COUNT(DISTINCT ps.player_id) as players_completed
            FROM player_sessions ps
            JOIN player_profiles pp ON ps.player_id = pp.player_id
            WHERE ps.start_time BETWEEN :start_date AND :end_date
                AND ps.level_reached IS NOT NULL
                AND ps.level_reached <= :max_level
                AND EXISTS (
                    SELECT 1 FROM player_sessions ps2 
                    WHERE ps2.player_id = ps.player_id 
                    AND ps2.level_reached > ps.level_reached
                )
            GROUP BY ps.level_reached
        )
        SELECT 
            lp.level_reached as level,
            lp.players_reached,
            COALESCE(lc.players_completed, 0) as players_completed,
            CASE 
                WHEN lp.players_reached > 0 THEN 
                    (lp.players_reached - COALESCE(lc.players_completed, 0))::DECIMAL / lp.players_reached
                ELSE 0 
            END as drop_off_rate,
            CASE 
                WHEN lp.players_reached > 0 THEN 
                    COALESCE(lc.players_completed, 0)::DECIMAL / lp.players_reached
                ELSE 0 
            END as completion_rate
        FROM level_progression lp
        LEFT JOIN level_completion lc ON lp.level_reached = lc.level_reached
        ORDER BY lp.level_reached
        """
        
        params = {
            'start_date': start_date,
            'end_date': end_date,
            'max_level': max_level
        }
        
        try:
            with self.db_manager.get_session() as session:
                result = session.execute(text(query), params)
                rows = result.fetchall()
                
                return [
                    DropOffAnalysisResult(
                        level=row.level,
                        players_reached=row.players_reached,
                        players_completed=row.players_completed,
                        drop_off_rate=float(row.drop_off_rate),
                        completion_rate=float(row.completion_rate)
                    )
                    for row in rows
                ]
        except Exception as e:
            logger.error(f"Error analyzing drop-off by level: {e}")
            raise
    
    def segment_players_by_behavior(
        self,
        start_date: date,
        end_date: date
    ) -> List[PlayerSegmentResult]:
        """Segment players based on behavior patterns.
        
        Args:
            start_date: Start date for segmentation analysis
            end_date: End date for segmentation analysis
            
        Returns:
            List of player segment results
        """
        query = """
        WITH player_metrics AS (
            SELECT 
                pp.player_id,
                pp.total_sessions,
                pp.total_playtime_minutes,
                CASE 
                    WHEN pp.total_sessions >= 20 AND pp.total_playtime_minutes >= 1200 THEN 'High Engagement'
                    WHEN pp.total_sessions >= 10 AND pp.total_playtime_minutes >= 600 THEN 'Medium Engagement'
                    WHEN pp.total_sessions >= 3 AND pp.total_playtime_minutes >= 180 THEN 'Low Engagement'
                    ELSE 'Minimal Engagement'
                END as segment,
                -- Calculate 7-day retention for each player
                CASE 
                    WHEN EXISTS (
                        SELECT 1 FROM player_sessions ps 
                        WHERE ps.player_id = pp.player_id 
                        AND ps.start_time >= pp.registration_date + INTERVAL '7 days'
                        AND ps.start_time < pp.registration_date + INTERVAL '8 days'
                    ) THEN 1.0 
                    ELSE 0.0 
                END as retained_day_7
            FROM player_profiles pp
            WHERE pp.registration_date BETWEEN :start_date AND :end_date
        )
        SELECT 
            segment,
            COUNT(*) as player_count,
            AVG(total_sessions) as avg_sessions,
            AVG(total_playtime_minutes) as avg_playtime,
            AVG(retained_day_7) as avg_retention_day_7
        FROM player_metrics
        GROUP BY segment
        ORDER BY 
            CASE segment
                WHEN 'High Engagement' THEN 1
                WHEN 'Medium Engagement' THEN 2
                WHEN 'Low Engagement' THEN 3
                WHEN 'Minimal Engagement' THEN 4
            END
        """
        
        params = {
            'start_date': start_date,
            'end_date': end_date
        }
        
        try:
            with self.db_manager.get_session() as session:
                result = session.execute(text(query), params)
                rows = result.fetchall()
                
                return [
                    PlayerSegmentResult(
                        segment=row.segment,
                        player_count=row.player_count,
                        avg_sessions=float(row.avg_sessions),
                        avg_playtime=float(row.avg_playtime),
                        avg_retention_day_7=float(row.avg_retention_day_7)
                    )
                    for row in rows
                ]
        except Exception as e:
            logger.error(f"Error segmenting players by behavior: {e}")
            raise
    
    def get_retention_by_segment(
        self,
        start_date: date,
        end_date: date,
        segment: str
    ) -> List[RetentionQueryResult]:
        """Get retention rates filtered by player segment.
        
        Args:
            start_date: Start date for analysis
            end_date: End date for analysis
            segment: Player segment to filter by
            
        Returns:
            List of retention results for the specified segment
        """
        query = """
        WITH player_segments AS (
            SELECT 
                pp.player_id,
                DATE(pp.registration_date) as cohort_date,
                pp.registration_date,
                CASE 
                    WHEN pp.total_sessions >= 20 AND pp.total_playtime_minutes >= 1200 THEN 'High Engagement'
                    WHEN pp.total_sessions >= 10 AND pp.total_playtime_minutes >= 600 THEN 'Medium Engagement'
                    WHEN pp.total_sessions >= 3 AND pp.total_playtime_minutes >= 180 THEN 'Low Engagement'
                    ELSE 'Minimal Engagement'
                END as segment
            FROM player_profiles pp
            WHERE DATE(pp.registration_date) BETWEEN :start_date AND :end_date
        ),
        filtered_cohorts AS (
            SELECT * FROM player_segments WHERE segment = :segment
        ),
        cohort_sizes AS (
            SELECT 
                cohort_date,
                COUNT(*) as cohort_size
            FROM filtered_cohorts
            GROUP BY cohort_date
        ),
        day_1_active AS (
            SELECT DISTINCT
                fc.cohort_date,
                fc.player_id
            FROM filtered_cohorts fc
            JOIN player_sessions ps ON fc.player_id = ps.player_id
            WHERE ps.start_time BETWEEN fc.registration_date + INTERVAL '1 day' 
                                   AND fc.registration_date + INTERVAL '2 days'
        ),
        day_7_active AS (
            SELECT DISTINCT
                fc.cohort_date,
                fc.player_id
            FROM filtered_cohorts fc
            JOIN player_sessions ps ON fc.player_id = ps.player_id
            WHERE ps.start_time BETWEEN fc.registration_date + INTERVAL '7 days' 
                                   AND fc.registration_date + INTERVAL '8 days'
        ),
        day_30_active AS (
            SELECT DISTINCT
                fc.cohort_date,
                fc.player_id
            FROM filtered_cohorts fc
            JOIN player_sessions ps ON fc.player_id = ps.player_id
            WHERE ps.start_time BETWEEN fc.registration_date + INTERVAL '30 days' 
                                   AND fc.registration_date + INTERVAL '31 days'
        )
        SELECT 
            cs.cohort_date,
            COALESCE(COUNT(DISTINCT d1.player_id)::DECIMAL / cs.cohort_size, 0) as day_1_retention,
            COALESCE(COUNT(DISTINCT d7.player_id)::DECIMAL / cs.cohort_size, 0) as day_7_retention,
            COALESCE(COUNT(DISTINCT d30.player_id)::DECIMAL / cs.cohort_size, 0) as day_30_retention,
            cs.cohort_size
        FROM cohort_sizes cs
        LEFT JOIN day_1_active d1 ON cs.cohort_date = d1.cohort_date
        LEFT JOIN day_7_active d7 ON cs.cohort_date = d7.cohort_date
        LEFT JOIN day_30_active d30 ON cs.cohort_date = d30.cohort_date
        GROUP BY cs.cohort_date, cs.cohort_size
        ORDER BY cs.cohort_date
        """
        
        params = {
            'start_date': start_date,
            'end_date': end_date,
            'segment': segment
        }
        
        try:
            with self.db_manager.get_session() as session:
                result = session.execute(text(query), params)
                rows = result.fetchall()
                
                return [
                    RetentionQueryResult(
                        cohort_date=row.cohort_date,
                        day_1_retention=float(row.day_1_retention),
                        day_7_retention=float(row.day_7_retention),
                        day_30_retention=float(row.day_30_retention),
                        cohort_size=row.cohort_size,
                        segment=segment
                    )
                    for row in rows
                ]
        except Exception as e:
            logger.error(f"Error getting retention by segment: {e}")
            raise
    
    def get_daily_active_users(
        self,
        start_date: date,
        end_date: date
    ) -> List[Tuple[date, int]]:
        """Get daily active user counts for a date range.
        
        Args:
            start_date: Start date for DAU calculation
            end_date: End date for DAU calculation
            
        Returns:
            List of tuples containing (date, dau_count)
        """
        query = """
        SELECT 
            DATE(ps.start_time) as activity_date,
            COUNT(DISTINCT ps.player_id) as dau_count
        FROM player_sessions ps
        WHERE DATE(ps.start_time) BETWEEN :start_date AND :end_date
        GROUP BY DATE(ps.start_time)
        ORDER BY activity_date
        """
        
        params = {
            'start_date': start_date,
            'end_date': end_date
        }
        
        try:
            with self.db_manager.get_session() as session:
                result = session.execute(text(query), params)
                rows = result.fetchall()
                
                return [(row.activity_date, row.dau_count) for row in rows]
        except Exception as e:
            logger.error(f"Error getting daily active users: {e}")
            raise
    
    def get_weekly_active_users(
        self,
        start_date: date,
        end_date: date
    ) -> List[Tuple[date, int]]:
        """Get weekly active user counts for a date range.
        
        Args:
            start_date: Start date for WAU calculation
            end_date: End date for WAU calculation
            
        Returns:
            List of tuples containing (week_start_date, wau_count)
        """
        query = """
        SELECT 
            DATE_TRUNC('week', ps.start_time)::date as week_start,
            COUNT(DISTINCT ps.player_id) as wau_count
        FROM player_sessions ps
        WHERE ps.start_time BETWEEN :start_date AND :end_date
        GROUP BY DATE_TRUNC('week', ps.start_time)
        ORDER BY week_start
        """
        
        params = {
            'start_date': start_date,
            'end_date': end_date
        }
        
        try:
            with self.db_manager.get_session() as session:
                result = session.execute(text(query), params)
                rows = result.fetchall()
                
                return [(row.week_start, row.wau_count) for row in rows]
        except Exception as e:
            logger.error(f"Error getting weekly active users: {e}")
            raise


# Global query engine instance
query_engine = RetentionQueryEngine()