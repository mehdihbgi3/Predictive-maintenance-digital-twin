SELECT 
    COUNT(*) as total_records,
    COUNT(DISTINCT serial_number) as unique_drives,
    COUNT(DISTINCT model) as unique_models,
    SUM(failure) as total_failures,
    ROUND(100.0 * SUM(failure) / COUNT(*), 4) as failure_rate_pct,
    MIN(date) as first_date,
    MAX(date) as last_date
FROM hard_drive.smart_data;

SELECT 
    DATE_TRUNC('month', date) as month,
    COUNT(*) as records,
    COUNT(DISTINCT serial_number) as active_drives,
    SUM(failure) as failures
FROM hard_drive.smart_data
GROUP BY DATE_TRUNC('month', date)
ORDER BY month;

SELECT 
    model,
    COUNT(DISTINCT serial_number) as total_drives,
    SUM(failure) as failures,
    ROUND(100.0 * SUM(failure) / COUNT(DISTINCT serial_number), 4) as failure_rate_pct
FROM hard_drive.smart_data
GROUP BY model
HAVING COUNT(DISTINCT serial_number) >= 100
ORDER BY failure_rate_pct DESC
LIMIT 20;

SELECT 
    serial_number,
    model,
    date as failure_date,
    smart_5_raw as reallocated_sectors,
    smart_187_raw as uncorrectable_errors,
    smart_197_raw as pending_sectors,
    smart_198_raw as offline_uncorrectable,
    smart_194_raw as temperature
FROM hard_drive.smart_data
WHERE failure = 1
ORDER BY date DESC
LIMIT 50;

SELECT 
    EXTRACT(DOW FROM date) as day_of_week,
    CASE EXTRACT(DOW FROM date)
        WHEN 0 THEN 'Sunday'
        WHEN 1 THEN 'Monday'
        WHEN 2 THEN 'Tuesday'
        WHEN 3 THEN 'Wednesday'
        WHEN 4 THEN 'Thursday'
        WHEN 5 THEN 'Friday'
        WHEN 6 THEN 'Saturday'
    END as day_name,
    SUM(failure) as failures
FROM hard_drive.smart_data
GROUP BY EXTRACT(DOW FROM date)
ORDER BY day_of_week;

SELECT 
    CASE WHEN failure = 1 THEN 'Failed' ELSE 'Healthy' END as status,
    ROUND(AVG(smart_5_raw), 2) as avg_reallocated_sectors,
    ROUND(AVG(smart_9_raw), 2) as avg_power_on_hours,
    ROUND(AVG(smart_187_raw), 2) as avg_uncorrectable_errors,
    ROUND(AVG(smart_194_raw), 2) as avg_temperature,
    ROUND(AVG(smart_197_raw), 2) as avg_pending_sectors,
    ROUND(AVG(smart_198_raw), 2) as avg_offline_uncorrectable
FROM hard_drive.smart_data
GROUP BY CASE WHEN failure = 1 THEN 'Failed' ELSE 'Healthy' END;

SELECT 
    serial_number,
    model,
    date,
    smart_5_raw as reallocated_sectors,
    smart_187_raw as uncorrectable_errors,
    smart_197_raw as pending_sectors
FROM hard_drive.smart_data
WHERE failure = 0
  AND (smart_5_raw > 100 OR smart_187_raw > 0 OR smart_197_raw > 0)
ORDER BY smart_5_raw DESC
LIMIT 100;

WITH drive_stats AS (
    SELECT 
        model,
        serial_number,
        MIN(date) as first_seen,
        MAX(date) as last_seen,
        MAX(failure) as failed
    FROM hard_drive.smart_data
    GROUP BY model, serial_number
)
SELECT 
    model,
    COUNT(*) as total_drives,
    SUM(failed) as failures,
    ROUND(AVG(last_seen - first_seen), 1) as avg_observation_days,
    ROUND(365.0 * SUM(failed) / (SUM(last_seen - first_seen) / COUNT(*)), 4) as annualized_failure_rate
FROM drive_stats
GROUP BY model
HAVING COUNT(*) >= 100 AND SUM(last_seen - first_seen) > 0
ORDER BY annualized_failure_rate DESC
LIMIT 20;

SELECT 
    CASE 
        WHEN capacity_bytes < 2000000000000 THEN '< 2TB'
        WHEN capacity_bytes < 4000000000000 THEN '2-4TB'
        WHEN capacity_bytes < 8000000000000 THEN '4-8TB'
        WHEN capacity_bytes < 12000000000000 THEN '8-12TB'
        ELSE '12TB+'
    END as capacity_range,
    COUNT(DISTINCT serial_number) as drives,
    SUM(failure) as failures,
    ROUND(100.0 * SUM(failure) / COUNT(DISTINCT serial_number), 4) as failure_rate_pct
FROM hard_drive.smart_data
GROUP BY 1
ORDER BY 1;

SELECT 
    date,
    COUNT(DISTINCT serial_number) as active_drives,
    SUM(failure) as daily_failures,
    ROUND(100000.0 * SUM(failure) / COUNT(DISTINCT serial_number), 4) as failures_per_100k
FROM hard_drive.smart_data
GROUP BY date
ORDER BY date;

WITH daily_failures AS (
    SELECT 
        date,
        SUM(failure) as failures
    FROM hard_drive.smart_data
    GROUP BY date
)
SELECT 
    date,
    failures,
    ROUND(AVG(failures) OVER (ORDER BY date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW), 2) as rolling_7d_avg
FROM daily_failures
ORDER BY date;