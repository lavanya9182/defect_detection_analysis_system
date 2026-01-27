import sqlite3
import pandas as pd
from datetime import datetime
import os

DB_NAME = "capsule_inspections.db"

def init_db():
    """Initialize the SQLite database and create the inspections table if it doesn't exist."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS inspections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_id TEXT,
            status TEXT,
            defect_type TEXT,
            score REAL,
            severity TEXT,
            timestamp DATETIME
        )
    ''')
    conn.commit()
    conn.close()

def log_inspection(image_id, status, defect_type, score, severity):
    """Log a single inspection result to the database."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    timestamp = datetime.now()
    c.execute('''
        INSERT INTO inspections (image_id, status, defect_type, score, severity, timestamp)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (image_id, status, defect_type, score, severity, timestamp))
    conn.commit()
    conn.close()

def fetch_data(period='All'):
    """
    Fetch inspection data for a specific period (Daily, Weekly, Monthly, or All).
    Returns a pandas DataFrame.
    """
    conn = sqlite3.connect(DB_NAME)
    
    query = "SELECT * FROM inspections"
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    if df.empty:
        return df

    # Convert timestamp to datetime objects
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Filter based on period
    now = datetime.now()
    if period == 'Daily':
        df = df[df['timestamp'].dt.date == now.date()]
    elif period == 'Weekly':
        # Last 7 days
        start_date = now - pd.Timedelta(days=7)
        df = df[df['timestamp'] >= start_date]
    elif period == 'Monthly':
        # Current month
        df = df[(df['timestamp'].dt.month == now.month) & (df['timestamp'].dt.year == now.year)]
    
    return df
