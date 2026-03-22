import psycopg2
import yaml
from pathlib import Path

config_path = Path("F:/predictive-maintenance-digital-twin/configs/config.yaml")
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)
db_config = config['database']

print("Testing database connection...")
print(f"   Host: {db_config['host']}")
print(f"   Database: {db_config['name']}")
print(f"   User: {db_config['user']}")

try:
    conn = psycopg2.connect(
        host=db_config['host'],
        port=db_config['port'],
        dbname=db_config['name'],
        user=db_config['user'],
        password=db_config['password']
    )
    
    cursor = conn.cursor()
    cursor.execute("SELECT version();")
    version = cursor.fetchone()[0]
    print(f"\nSUCCESS! Connected to PostgreSQL")
    print(f"   Version: {version[:50]}...")
    
    cursor.execute("""
        SELECT table_schema, table_name 
        FROM information_schema.tables 
        WHERE table_schema IN ('hard_drive', 'features')
    """)
    tables = cursor.fetchall()
    print(f"\nTables found: {len(tables)}")
    for schema, table in tables:
        print(f"   - {schema}.{table}")
    
    cursor.close()
    conn.close()
    
except Exception as e:
    print(f"\nCONNECTION FAILED!")
    print(f"   Error: {str(e)}")
    print("\nCheck:")
    print("   1. Is PostgreSQL running?")
    print("   2. Is your password correct in config.yaml?")
    print("   3. Does the database 'predictive_maintenance' exist?")
