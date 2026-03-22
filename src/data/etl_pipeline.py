import os
import glob
from pathlib import Path
import pandas as pd
import psycopg2
from io import StringIO
import yaml
import time
import numpy as np


def load_config():
    config_path = Path("F:/predictive-maintenance-digital-twin/configs/config.yaml")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


class DatabaseManager:
    def __init__(self, config):
        self.config = config['database']
        self.conn = None
        
    def connect(self):
        self.conn = psycopg2.connect(
            host=self.config['host'],
            port=self.config['port'],
            dbname=self.config['name'],
            user=self.config['user'],
            password=self.config['password']
        )
        cur = self.conn.cursor()
        cur.execute('SET search_path TO hard_drive, public')
        self.conn.commit()
        cur.close()
        print("[OK] Connected to PostgreSQL")
        return self.conn
    
    def close(self):
        if self.conn:
            self.conn.close()
            print("[OK] Database connection closed")
    
    def bulk_insert_manual(self, df):
        lines = []
        for _, row in df.iterrows():
            values = []
            for col in df.columns:
                val = row[col]
                if pd.isna(val):
                    values.append('\\N')
                elif isinstance(val, float):
                    values.append(str(int(val)))
                else:
                    values.append(str(val))
            lines.append('\t'.join(values))
        
        buffer = StringIO('\n'.join(lines))
        
        cursor = self.conn.cursor()
        try:
            cursor.copy_from(buffer, 'smart_data', sep='\t', null='\\N', columns=df.columns.tolist())
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            raise e
        finally:
            cursor.close()


class BackblazeProcessor:
    
    def __init__(self, config):
        self.chunk_size = config['etl']['chunk_size']
    
    def find_csv_files(self, data_path):
        csv_files = glob.glob(os.path.join(data_path, "**", "*.csv"), recursive=True)
        csv_files = sorted(set(csv_files))
        print(f"[INFO] Found {len(csv_files)} CSV files")
        return csv_files
    
    def process_chunk(self, chunk):
        result = pd.DataFrame()
        
        result['date'] = pd.to_datetime(chunk['date']).dt.strftime('%Y-%m-%d')
        
        result['serial_number'] = chunk['serial_number'].astype(str)
        result['model'] = chunk['model'].astype(str) if 'model' in chunk.columns else ''
        
        result['capacity_bytes'] = pd.to_numeric(chunk['capacity_bytes'], errors='coerce')
        result['failure'] = pd.to_numeric(chunk['failure'], errors='coerce').fillna(0)
        
        smart_cols = ['smart_5_raw', 'smart_9_raw', 'smart_187_raw', 'smart_188_raw',
                      'smart_194_raw', 'smart_197_raw', 'smart_198_raw', 'smart_199_raw',
                      'smart_241_raw', 'smart_242_raw']
        
        for col in smart_cols:
            if col in chunk.columns:
                result[col] = pd.to_numeric(chunk[col], errors='coerce')
            else:
                result[col] = np.nan
        
        return result
    
    def process_file(self, filepath, db_manager):
        filename = os.path.basename(filepath)
        file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
        
        print(f"  Processing: {filename} ({file_size_mb:.1f} MB)", end="", flush=True)
        
        rows_inserted = 0
        
        try:
            for chunk in pd.read_csv(filepath, chunksize=self.chunk_size, low_memory=False):
                processed = self.process_chunk(chunk)
                db_manager.bulk_insert_manual(processed)
                rows_inserted += len(processed)
            
            print(f" -> {rows_inserted:,} rows [OK]")
            return rows_inserted
            
        except Exception as e:
            print(f" -> [ERROR] {str(e)[:60]}")
            return 0


def run_etl():
    start_time = time.time()
    
    print("-" * 60)
    print("PREDICTIVE MAINTENANCE ETL PIPELINE")
    print("-" * 60)
    
    config = load_config()
    db_manager = DatabaseManager(config)
    processor = BackblazeProcessor(config)
    
    db_manager.connect()
    
    data_path = config['paths']['raw_data']
    csv_files = processor.find_csv_files(data_path)
    
    if not csv_files:
        print("[ERROR] No CSV files found!")
        return
    
    print(f"\nProcessing {len(csv_files)} files...")
    print("-" * 60)
    
    total_rows = 0
    successful = 0
    
    for i, filepath in enumerate(csv_files, 1):
        print(f"[{i}/{len(csv_files)}]", end=" ")
        rows = processor.process_file(filepath, db_manager)
        total_rows += rows
        if rows > 0:
            successful += 1
    
    db_manager.close()
    
    elapsed = time.time() - start_time



if __name__ == "__main__":
    run_etl()
