-- Player Retention Analysis Queries

-- 1. Daily Active Users (DAU)
CREATE VIEW daily_active_users AS
SELECT 
    DATE(session_date) as date,
    COUNT(DISTINCT player_id) as daily_active_users
FROM sessions
GROUP BY DATE(session_date)
ORDER BY date;

-- 2. Weekly Retention Rate
CREATE VIEW weekly_retention AS
WITH player_weeks AS (
    SELECT 
        player_id,
        p.registration_date,
        DATE(s.session_date) as session_date,
        CAST((julianday(s.session_date) - julianday(p.registration_date)) / 7 AS INTEGER) as week_number
    FROM sessions s
    JOIN players p ON s.player_id = p.player_id
),
cohort_sizes AS (
    SELECT 
        DATE(registration_date) as cohort_date,
        COUNT(DISTINCT player_id) as cohort_size
    FROM players
    GROUP BY DATE(registration_date)
),
retention_data AS (
    SELECT 
        DATE(pw.registration_date) as cohort_date,
        pw.week_number,
        COUNT(DISTINCT pw.player_id) as retained_players
    FROM player_weeks pw
    WHERE pw.week_number <= 8  -- Track up to 8 weeks
    GROUP BY DATE(pw.registration_date), pw.week_number
)
SELECT 
    rd.cohort_date,
    rd.week_number,
    rd.retained_players,
    cs.cohort_size,
    ROUND(100.0 * rd.retained_players / cs.cohort_size, 2) as retention_rate
FROM retention_data rd
JOIN cohort_sizes cs ON rd.cohort_date = cs.cohort_date
ORDER BY rd.cohort_date, rd.week_number;

-- 3. Player Lifetime Value (LTV) Analysis
CREATE VIEW player_ltv AS
SELECT 
    player_id,
    COUNT(*) as total_sessions,
    SUM(session_length_minutes) as total_playtime,
    SUM(purchases_made) as total_purchases,
    SUM(revenue) as total_revenue,
    AVG(session_length_minutes) as avg_session_length,
    MAX(level_reached) as max_level,
    MIN(DATE(session_date)) as first_session,
    MAX(DATE(session_date)) as last_session,
    julianday(MAX(session_date)) - julianday(MIN(session_date)) + 1 as lifetime_days
FROM sessions
GROUP BY player_id;

-- 4. Churn Analysis - Players who haven't played in 30+ days
CREATE VIEW churned_players AS
WITH player_last_activity AS (
    SELECT 
        player_id,
        MAX(DATE(session_date)) as last_activity_date,
        julianday('now') - julianday(MAX(session_date)) as days_since_last_activity
    FROM sessions
    GROUP BY player_id
)
SELECT 
    pla.*,
    ltv.total_sessions,
    ltv.total_revenue,
    ltv.lifetime_days,
    CASE 
        WHEN days_since_last_activity > 30 THEN 1 
        ELSE 0 
    END as is_churned
FROM player_last_activity pla
JOIN player_ltv ltv ON pla.player_id = ltv.player_id;

-- 5. Session Patterns - Peak hours and days
CREATE VIEW session_patterns AS
SELECT 
    strftime('%H', session_date) as hour_of_day,
    strftime('%w', session_date) as day_of_week,
    COUNT(*) as session_count,
    AVG(session_length_minutes) as avg_session_length,
    SUM(revenue) as total_revenue
FROM sessions
GROUP BY strftime('%H', session_date), strftime('%w', session_date)
ORDER BY session_count DESC;