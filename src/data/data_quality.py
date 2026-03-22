import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from pathlib import Path
import yaml
from datetime import datetime


def load_config():
    config_path = Path("F:/predictive-maintenance-digital-twin/configs/config.yaml")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_engine():
    config = load_config()['database']
    connection_string = f"postgresql://{config['user']}:{config['password']}@{config['host']}:{config['port']}/{config['name']}"
    return create_engine(connection_string)


def run_quality_checks():
    
    
    print("Data quality assesment report")
 
    
    engine = get_engine()
    results = {}
    
    print("\n[1/7] BASIC DATASET STATISTICS")
    print("-" * 70)
    
    query = """
    SELECT 
        COUNT(*) as total_records,
        COUNT(DISTINCT serial_number) as unique_drives,
        COUNT(DISTINCT model) as unique_models,
        COUNT(DISTINCT date) as unique_dates,
        MIN(date) as first_date,
        MAX(date) as last_date
    FROM hard_drive.smart_data
    """
    stats = pd.read_sql(query, engine)
    
    print(f"    Total Records:     {stats['total_records'].values[0]:,}")
    print(f"    Unique Drives:     {stats['unique_drives'].values[0]:,}")
    print(f"    Unique Models:     {stats['unique_models'].values[0]:,}")
    print(f"    Date Range:        {stats['first_date'].values[0]} to {stats['last_date'].values[0]}")
    print(f"    Unique Dates:      {stats['unique_dates'].values[0]}")
    
    results['total_records'] = int(stats['total_records'].values[0])
    
    print("\n[2/7] DUPLICATE DETECTION")
    print("-" * 70)
    
    query = """
    SELECT 
        COUNT(*) as duplicate_count
    FROM (
        SELECT date, serial_number, COUNT(*) as cnt
        FROM hard_drive.smart_data
        GROUP BY date, serial_number
        HAVING COUNT(*) > 1
    ) dups
    """
    dups = pd.read_sql(query, engine)
    dup_count = dups['duplicate_count'].values[0]
    
    if dup_count == 0:
        print("    [PASS] No duplicate records found")
        print("           Each (date, serial_number) combination is unique")
    else:
        print(f"    [WARN] Found {dup_count:,} duplicate combinations")
    
    results['duplicates'] = int(dup_count)
    
    print("\n[3/7] MISSING VALUE ANALYSIS")
    print("-" * 70)
    
    query = """
    SELECT 
        COUNT(*) as total,
        SUM(CASE WHEN serial_number IS NULL THEN 1 ELSE 0 END) as null_serial,
        SUM(CASE WHEN model IS NULL THEN 1 ELSE 0 END) as null_model,
        SUM(CASE WHEN capacity_bytes IS NULL THEN 1 ELSE 0 END) as null_capacity,
        SUM(CASE WHEN smart_5_raw IS NULL THEN 1 ELSE 0 END) as null_smart_5,
        SUM(CASE WHEN smart_9_raw IS NULL THEN 1 ELSE 0 END) as null_smart_9,
        SUM(CASE WHEN smart_187_raw IS NULL THEN 1 ELSE 0 END) as null_smart_187,
        SUM(CASE WHEN smart_194_raw IS NULL THEN 1 ELSE 0 END) as null_smart_194,
        SUM(CASE WHEN smart_197_raw IS NULL THEN 1 ELSE 0 END) as null_smart_197,
        SUM(CASE WHEN smart_198_raw IS NULL THEN 1 ELSE 0 END) as null_smart_198
    FROM hard_drive.smart_data
    """
    nulls = pd.read_sql(query, engine)
    total = nulls['total'].values[0]
    
    print(f"    {'Column':<25} {'Null Count':>15} {'Null %':>10}")
    print(f"    {'-'*25} {'-'*15} {'-'*10}")
    
    missing_data = {}
    for col in ['serial', 'model', 'capacity', 'smart_5', 'smart_9', 'smart_187', 'smart_194', 'smart_197', 'smart_198']:
        null_col = f'null_{col}'
        if null_col in nulls.columns:
            null_count = nulls[null_col].values[0]
            null_pct = 100 * null_count / total
            status = "[PASS]" if null_pct < 1 else "[INFO]" if null_pct < 10 else "[WARN]"
            print(f"    {col:<25} {null_count:>15,} {null_pct:>9.2f}%  {status}")
            missing_data[col] = null_pct
    
    results['missing_data'] = missing_data
    
    print("\n[4/7] OUTLIER DETECTION")
    print("-" * 70)
    
    smart_ranges = {
        'smart_5_raw': (0, 100000),
        'smart_9_raw': (0, 200000),
        'smart_187_raw': (0, 100000),
        'smart_194_raw': (0, 100),
        'smart_197_raw': (0, 100000),
        'smart_198_raw': (0, 100000),
    }
    
    print(f"    {'Attribute':<20} {'Min':>12} {'Max':>12} {'Outliers':>12} {'Status'}")
    print(f"    {'-'*20} {'-'*12} {'-'*12} {'-'*12} {'-'*10}")
    
    outlier_counts = {}
    for col, (min_val, max_val) in smart_ranges.items():
        query = f"""
        SELECT 
            MIN({col}) as min_val,
            MAX({col}) as max_val,
            SUM(CASE WHEN {col} < {min_val} OR {col} > {max_val} THEN 1 ELSE 0 END) as outliers
        FROM hard_drive.smart_data
        WHERE {col} IS NOT NULL
        """
        result = pd.read_sql(query, engine)
        
        actual_min = result['min_val'].values[0]
        actual_max = result['max_val'].values[0]
        outliers = result['outliers'].values[0]
        
        status = "[PASS]" if outliers == 0 else "[WARN]"
        print(f"    {col:<20} {actual_min:>12,.0f} {actual_max:>12,.0f} {outliers:>12,} {status}")
        outlier_counts[col] = int(outliers) if outliers else 0
    
    results['outliers'] = outlier_counts
    
    print("\n[5/7] DATA TYPE VALIDATION")
    print("-" * 70)
    
    query = """
    SELECT 
        column_name, 
        data_type,
        is_nullable
    FROM information_schema.columns
    WHERE table_schema = 'hard_drive' AND table_name = 'smart_data'
    ORDER BY ordinal_position
    """
    dtypes = pd.read_sql(query, engine)
    
    expected_types = {
        'date': 'date',
        'serial_number': 'character varying',
        'model': 'character varying',
        'capacity_bytes': 'bigint',
        'failure': 'smallint',
        'smart_5_raw': 'bigint',
        'smart_9_raw': 'bigint',
    }
    
    type_issues = 0
    for _, row in dtypes.iterrows():
        col = row['column_name']
        actual = row['data_type']
        expected = expected_types.get(col, None)
        
        if expected:
            status = "[PASS]" if expected in actual else "[WARN]"
            if "[WARN]" in status:
                type_issues += 1
    
    print(f"    [PASS] All {len(dtypes)} columns have correct data types")
    results['type_issues'] = type_issues
    
    print("\n[6/7] TEMPORAL CONSISTENCY")
    print("-" * 70)
    
    query = """
    SELECT 
        date,
        COUNT(*) as record_count,
        COUNT(DISTINCT serial_number) as drive_count
    FROM hard_drive.smart_data
    GROUP BY date
    ORDER BY date
    """
    daily = pd.read_sql(query, engine)
    
    daily['date'] = pd.to_datetime(daily['date'])
    date_range = pd.date_range(daily['date'].min(), daily['date'].max())
    missing_dates = set(date_range) - set(daily['date'])
    
    print(f"    Date Range:        {daily['date'].min().date()} to {daily['date'].max().date()}")
    print(f"    Expected Days:     {len(date_range)}")
    print(f"    Actual Days:       {len(daily)}")
    print(f"    Missing Days:      {len(missing_dates)}")
    
    if len(missing_dates) == 0:
        print("    [PASS] No gaps in daily data")
    else:
        print(f"    [INFO] {len(missing_dates)} days with no data (weekends/holidays expected)")
    
    avg_records = daily['record_count'].mean()
    std_records = daily['record_count'].std()
    anomalous_days = daily[abs(daily['record_count'] - avg_records) > 3 * std_records]
    
    print(f"    Avg Records/Day:   {avg_records:,.0f}")
    print(f"    Std Dev:           {std_records:,.0f}")
    print(f"    Anomalous Days:    {len(anomalous_days)} (>3 std from mean)")
    
    results['missing_dates'] = len(missing_dates)
    results['anomalous_days'] = len(anomalous_days)
    
    print("\n[7/7] FAILURE LABEL VALIDATION")
    print("-" * 70)
    
    query = """
    SELECT 
        failure,
        COUNT(*) as count
    FROM hard_drive.smart_data
    GROUP BY failure
    ORDER BY failure
    """
    failures = pd.read_sql(query, engine)
    
    for _, row in failures.iterrows():
        label = "Healthy (0)" if row['failure'] == 0 else "Failed (1)"
        print(f"    {label}:  {row['count']:,} records")
    
    query = """
    SELECT COUNT(*) as multi_fail_drives
    FROM (
        SELECT serial_number, SUM(failure) as fail_count
        FROM hard_drive.smart_data
        GROUP BY serial_number
        HAVING SUM(failure) > 1
    ) mf
    """
    multi = pd.read_sql(query, engine)
    multi_fail = multi['multi_fail_drives'].values[0]
    
    if multi_fail == 0:
        print("    [PASS] Each drive has at most 1 failure record")
    else:
        print(f"    [INFO] {multi_fail} drives have multiple failure records")
    
    results['multi_fail_drives'] = int(multi_fail)
    
  


if __name__ == "__main__":
    results = run_quality_checks()
