import sqlite3
import pandas as pd
from typing import List, Dict, Any, Optional
import json

class SQLiteTool:
    def __init__(self, db_path: str = "data/northwind.sqlite"):
        self.db_path = db_path
        self.conn = None
        self._connect()
    
    def _connect(self):
        """Establish database connection"""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
    
    def get_schema(self, table_name: Optional[str] = None) -> str:
        """Get database schema information"""
        query = """
        SELECT name, sql FROM sqlite_master 
        WHERE type='table' OR type='view'
        """
        if table_name:
            query += f" AND name = '{table_name}'"
        
        schema_info = []
        cursor = self.conn.execute(query)
        for row in cursor:
            schema_info.append(f"Table/View: {row['name']}\nSQL: {row['sql']}\n")
        
        return "\n".join(schema_info)
    
    def get_table_names(self) -> List[str]:
        """Get list of all table names"""
        cursor = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' OR type='view'"
        )
        return [row['name'] for row in cursor]
    
    def execute_query(self, query: str) -> Dict[str, Any]:
        """Execute SQL query and return results"""
        try:
            df = pd.read_sql_query(query, self.conn)
            return {
                "success": True,
                "columns": list(df.columns),
                "rows": df.to_dict('records'),
                "row_count": len(df)
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "columns": [],
                "rows": [],
                "row_count": 0
            }
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()