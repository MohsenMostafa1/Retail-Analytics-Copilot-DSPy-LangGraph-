#!/usr/bin/env python3
import sqlite3
import pandas as pd

def test_database():
    try:
        # Test database connection
        conn = sqlite3.connect('data/northwind.sqlite')
        cursor = conn.cursor()
        
        # Test if tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' OR type='view'")
        tables = cursor.fetchall()
        print("Available tables/views:")
        for table in tables:
            print(f"  - {table[0]}")
        
        # Test a simple query
        df = pd.read_sql_query("SELECT COUNT(*) as count FROM orders", conn)
        print(f"Number of orders: {df['count'].iloc[0]}")
        
        conn.close()
        print("Database test: SUCCESS")
        
    except Exception as e:
        print(f"Database test: FAILED - {e}")

if __name__ == "__main__":
    test_database()
