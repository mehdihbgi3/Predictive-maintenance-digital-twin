import psycopg2
from io import StringIO
import pandas as pd

conn = psycopg2.connect(
    host='localhost',
    port=5432,
    dbname='predictive_maintenance',
    user='postgres',
    password='postgres'
)
cur = conn.cursor()

cur.execute('SET search_path TO hard_drive, public')

data = {
    'date': ['2025-01-04'],
    'serial_number': ['TEST999'],
    'model': ['TestModel4'],
    'capacity_bytes': [4000000],
    'failure': [0],
    'smart_5_raw': [None],
    'smart_9_raw': [200],
    'smart_187_raw': [None],
    'smart_188_raw': [None],
    'smart_194_raw': [40],
    'smart_197_raw': [None],
    'smart_198_raw': [None],
    'smart_199_raw': [None],
    'smart_241_raw': [None],
    'smart_242_raw': [None]
}
df = pd.DataFrame(data)
df['date'] = pd.to_datetime(df['date']).dt.date

print("Testing bulk insert with search_path fix...")

buffer = StringIO()
df.to_csv(buffer, index=False, header=False, sep='\t', na_rep='\\N')
buffer.seek(0)

try:
    cur.copy_from(
        buffer,
        'smart_data',
        sep='\t',
        null='\\N',
        columns=df.columns.tolist()
    )
    conn.commit()
except Exception as e:
    conn.rollback()
    print(f"Bulk insert Failed: {e}")

cur.execute('SELECT count(*) FROM hard_drive.smart_data')
print(f"Total rows in table: {cur.fetchone()[0]}")
conn.close()
