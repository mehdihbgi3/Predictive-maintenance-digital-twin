import psycopg2
import pandas as pd
from pathlib import Path
import yaml
from sqlalchemy import create_engine


def load_config():
    config_path = Path("F:/predictive-maintenance-digital-twin/configs/config.yaml")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_engine():
    config = load_config()['database']
    connection_string = f"postgresql://{config['user']}:{config['password']}@{config['host']}:{config['port']}/{config['name']}"
    return create_engine(connection_string)


def run_query(query):
    engine = get_engine()
    return pd.read_sql(query, engine)


def get_dataset_overview():
    query = """
    SELECT 
        COUNT(*) as total_records,
        COUNT(DISTINCT serial_number) as unique_drives,
        COUNT(DISTINCT model) as unique_models,
        SUM(failure) as total_failures,
        ROUND(100.0 * SUM(failure) / COUNT(*), 6) as failure_rate_pct,
        MIN(date) as first_date,
        MAX(date) as last_date
    FROM hard_drive.smart_data
    """
    return run_query(query)


def get_failure_by_model(min_drives=100, limit=20):
    query = f"""
    SELECT 
        model,
        COUNT(DISTINCT serial_number) as total_drives,
        SUM(failure) as failures,
        ROUND(100.0 * SUM(failure) / COUNT(DISTINCT serial_number), 4) as failure_rate_pct
    FROM hard_drive.smart_data
    GROUP BY model
    HAVING COUNT(DISTINCT serial_number) >= {min_drives}
    ORDER BY failure_rate_pct DESC
    LIMIT {limit}
    """
    return run_query(query)


def get_smart_comparison():
    query = """
    SELECT 
        CASE WHEN failure = 1 THEN 'Failed' ELSE 'Healthy' END as status,
        ROUND(AVG(smart_5_raw)::numeric, 2) as avg_reallocated_sectors,
        ROUND(AVG(smart_9_raw)::numeric, 2) as avg_power_on_hours,
        ROUND(AVG(smart_187_raw)::numeric, 2) as avg_uncorrectable_errors,
        ROUND(AVG(smart_194_raw)::numeric, 2) as avg_temperature,
        ROUND(AVG(smart_197_raw)::numeric, 2) as avg_pending_sectors,
        ROUND(AVG(smart_198_raw)::numeric, 2) as avg_offline_uncorrectable
    FROM hard_drive.smart_data
    GROUP BY CASE WHEN failure = 1 THEN 'Failed' ELSE 'Healthy' END
    """
    return run_query(query)


def get_daily_failures():
    query = """
    SELECT 
        date,
        COUNT(DISTINCT serial_number) as active_drives,
        SUM(failure) as daily_failures
    FROM hard_drive.smart_data
    GROUP BY date
    ORDER BY date
    """
    return run_query(query)


def get_capacity_analysis():
    query = """
    SELECT 
        CASE 
            WHEN capacity_bytes < 2000000000000 THEN '< 2TB'
            WHEN capacity_bytes < 4000000000000 THEN '2-4TB'
            WHEN capacity_bytes < 8000000000000 THEN '4-8TB'
            WHEN capacity_bytes < 12000000000000 THEN '8-12TB'
            WHEN capacity_bytes < 16000000000000 THEN '12-16TB'
            ELSE '16TB+'
        END as capacity_range,
        COUNT(DISTINCT serial_number) as drives,
        SUM(failure) as failures,
        ROUND(100.0 * SUM(failure) / COUNT(DISTINCT serial_number), 4) as failure_rate_pct
    FROM hard_drive.smart_data
    WHERE capacity_bytes IS NOT NULL
    GROUP BY 1
    ORDER BY MIN(capacity_bytes)
    """
    return run_query(query)


def print_report():
    
    print("       PREDICTIVE MAINTENANCE - DATA ANALYTICS REPORT")
    print("-" * 70)
    
    print("\n[1] DATASET STATISTICS")
    print("-" * 70)
    overview = get_dataset_overview()
    print(f"    Total Records:     {overview['total_records'].values[0]:,}")
    print(f"    Unique Drives:     {overview['unique_drives'].values[0]:,}")
    print(f"    Unique Models:     {overview['unique_models'].values[0]:,}")
    print(f"    Total Failures:    {overview['total_failures'].values[0]:,}")
    print(f"    Failure Rate:      {overview['failure_rate_pct'].values[0]:.4f}%")
    print(f"    Date Range:        {overview['first_date'].values[0]} to {overview['last_date'].values[0]}")
    
    print("\n[2] TOP 10 MODELS BY FAILURE RATE")
    print("-" * 70)
    models = get_failure_by_model(limit=10)
    print(models.to_string(index=False))
    
    print("\n[3] SMART VALUES: FAILED vs HEALTHY DRIVES")
    print("-" * 70)
    smart = get_smart_comparison()
    print(smart.to_string(index=False))
    
    print("\n[4] FAILURE RATE BY CAPACITY")
    print("-" * 70)
    capacity = get_capacity_analysis()
    print(capacity.to_string(index=False))
    


if __name__ == "__main__":
    print_report()
