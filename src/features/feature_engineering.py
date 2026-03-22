import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from pathlib import Path
import yaml
import time
import warnings
warnings.filterwarnings('ignore')


def load_config():
    config_path = Path("F:/predictive-maintenance-digital-twin/configs/config.yaml")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_engine():
    config = load_config()['database']
    connection_string = f"postgresql://{config['user']}:{config['password']}@{config['host']}:{config['port']}/{config['name']}"
    return create_engine(connection_string)


def get_failed_drives():
    query = """
    SELECT DISTINCT serial_number, date as failure_date
    FROM hard_drive.smart_data
    WHERE failure = 1
    """
    engine = get_engine()
    return pd.read_sql(query, engine)


def get_drive_data_batch(serial_numbers):
    engine = get_engine()
    serials_str = "','".join(serial_numbers)
    
    query = f"""
    SELECT 
        date, serial_number, model, capacity_bytes, failure,
        smart_5_raw, smart_9_raw, smart_187_raw, smart_188_raw,
        smart_194_raw, smart_197_raw, smart_198_raw, smart_199_raw,
        smart_241_raw, smart_242_raw
    FROM hard_drive.smart_data
    WHERE serial_number IN ('{serials_str}')
    ORDER BY serial_number, date
    """
    return pd.read_sql(query, engine)


def create_features_for_drive(drive_df, failure_date=None):
    df = drive_df.sort_values('date').copy()
    
    if len(df) < 2:
        return None
    
    features = pd.DataFrame()
    features['date'] = df['date']
    features['serial_number'] = df['serial_number']
    features['model'] = df['model'].iloc[0] if 'model' in df.columns else 'unknown'
    
    smart_cols = ['smart_5_raw', 'smart_9_raw', 'smart_187_raw', 'smart_188_raw',
                  'smart_194_raw', 'smart_197_raw', 'smart_198_raw', 'smart_199_raw']
    
    for col in smart_cols:
        if col in df.columns:
            features[col] = df[col].values
        else:
            features[col] = 0
    
    for col in ['smart_5_raw', 'smart_187_raw', 'smart_197_raw', 'smart_198_raw']:
        if col in df.columns:
            series = df[col].fillna(0)
            features[f'{col}_mean_7d'] = series.rolling(7, min_periods=1).mean().values
            features[f'{col}_std_7d'] = series.rolling(7, min_periods=1).std().fillna(0).values
            features[f'{col}_max_7d'] = series.rolling(7, min_periods=1).max().values
            features[f'{col}_delta_7d'] = series.diff(7).fillna(0).values
    
    for col in ['smart_5_raw', 'smart_187_raw', 'smart_197_raw']:
        if col in df.columns:
            series = df[col].fillna(0)
            features[f'{col}_mean_30d'] = series.rolling(30, min_periods=1).mean().values
    
    if 'smart_194_raw' in df.columns:
        temp = df['smart_194_raw'].fillna(0)
        features['temp_mean_7d'] = temp.rolling(7, min_periods=1).mean().values
        features['temp_max_7d'] = temp.rolling(7, min_periods=1).max().values
    
    features['power_on_hours'] = df['smart_9_raw'].fillna(0).values
    
    first_date = pd.to_datetime(df['date']).min()
    features['age_days'] = (pd.to_datetime(df['date']) - first_date).dt.days
    
    cap = df['capacity_bytes'].iloc[0]
    features['capacity_tb'] = cap / 1e12 if pd.notna(cap) and cap > 0 else 0
    
    if failure_date is not None:
        failure_date = pd.to_datetime(failure_date)
        features['days_to_failure'] = (failure_date - pd.to_datetime(df['date'])).dt.days
        features['will_fail_30d'] = (features['days_to_failure'] <= 30) & (features['days_to_failure'] >= 0)
        features['will_fail_60d'] = (features['days_to_failure'] <= 60) & (features['days_to_failure'] >= 0)
        features['failed'] = 1
    else:
        features['days_to_failure'] = -1
        features['will_fail_30d'] = False
        features['will_fail_60d'] = False
        features['failed'] = 0
    
    return features


def create_training_dataset(n_healthy_drives=3000, save_path=None):
    print("[1/4] Getting failed drives...")
    start_time = time.time()
    engine = get_engine()
    
    failed_drives = get_failed_drives()
    print(f"Found {len(failed_drives)} failed drives")
    
    print(f"[2/4] Sampling {n_healthy_drives} healthy drives...")
    healthy_query = f"""
    SELECT serial_number FROM (
        SELECT DISTINCT serial_number
        FROM hard_drive.smart_data
        WHERE serial_number NOT IN (
            SELECT DISTINCT serial_number FROM hard_drive.smart_data WHERE failure = 1
        )
    ) AS healthy
    ORDER BY RANDOM()
    LIMIT {n_healthy_drives}
    """
    healthy_drives = pd.read_sql(healthy_query, engine)
    print(f"Sampled {len(healthy_drives)} healthy drives")
    
    all_features = []
    
    print("[3/4] Creating features for failed drives...")
    failed_serials = failed_drives['serial_number'].tolist()
    failed_dates = dict(zip(failed_drives['serial_number'], failed_drives['failure_date']))
    
    batch_size = 100
    for i in range(0, len(failed_serials), batch_size):
        batch = failed_serials[i:i+batch_size]
        
        if (i + batch_size) % 500 == 0 or i == 0:
            print(f"Processing {i+1}-{min(i+batch_size, len(failed_serials))}/{len(failed_serials)}")
        
        try:
            batch_data = get_drive_data_batch(batch)
            
            for serial in batch:
                drive_data = batch_data[batch_data['serial_number'] == serial]
                if len(drive_data) > 1:
                    features = create_features_for_drive(drive_data, failed_dates.get(serial))
                    if features is not None:
                        all_features.append(features)
        except Exception as e:
            print(f"Error in batch: {str(e)[:50]}")
            continue
    
    print("[4/4] Creating features for healthy drives...")
    healthy_serials = healthy_drives['serial_number'].tolist()
    
    for i in range(0, len(healthy_serials), batch_size):
        batch = healthy_serials[i:i+batch_size]
        
        if (i + batch_size) % 500 == 0 or i == 0:
            print(f"Processing {i+1}-{min(i+batch_size, len(healthy_serials))}/{len(healthy_serials)}")
        
        try:
            batch_data = get_drive_data_batch(batch)
            
            for serial in batch:
                drive_data = batch_data[batch_data['serial_number'] == serial]
                if len(drive_data) > 1:
                    features = create_features_for_drive(drive_data, failure_date=None)
                    if features is not None:
                        all_features.append(features)
        except Exception as e:
            print(f"Error in batch: {str(e)[:50]}")
            continue
    
    print("Combining features...")
    dataset = pd.concat(all_features, ignore_index=True)
    
    elapsed = time.time() - start_time
    print(f"Total samples: {len(dataset):,}")
    print(f"Features created: {len(dataset.columns)}")
    print(f"Time elapsed: {elapsed/60:.1f} minutes")
    print(f"Positive (will fail): {dataset['will_fail_30d'].sum():,}")
    print(f"Negative (healthy): {(~dataset['will_fail_30d']).sum():,}")
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        dataset.to_parquet(save_path, index=False)
        print(f"Saved to: {save_path}")
    
    return dataset


if __name__ == "__main__":
    save_path = "F:/predictive-maintenance-digital-twin/data/features/training_data.parquet"
    dataset = create_training_dataset(n_healthy_drives=3000, save_path=save_path)
