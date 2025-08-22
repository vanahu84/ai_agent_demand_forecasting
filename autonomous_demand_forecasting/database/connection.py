"""
Database connection and management utilities.
"""

import sqlite3
import os
from pathlib import Path
from typing import Optional, Dict, Any, List
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)


class DatabaseConnection:
    """Database connection manager for autonomous demand forecasting system."""
    
    def __init__(self, db_path: str = "autonomous_demand_forecasting.db"):
        """Initialize database connection.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._connection: Optional[sqlite3.Connection] = None
        
    def connect(self) -> sqlite3.Connection:
        """Establish database connection with optimized settings."""
        if self._connection is None:
            self._connection = sqlite3.connect(
                self.db_path,
                check_same_thread=False,
                timeout=30.0
            )
            
            # Enable foreign key constraints
            self._connection.execute("PRAGMA foreign_keys = ON")
            
            # Optimize for performance
            self._connection.execute("PRAGMA journal_mode = WAL")
            self._connection.execute("PRAGMA synchronous = NORMAL")
            self._connection.execute("PRAGMA cache_size = 10000")
            self._connection.execute("PRAGMA temp_store = MEMORY")
            
            # Set row factory for dict-like access
            self._connection.row_factory = sqlite3.Row
            
        return self._connection
    
    def close(self):
        """Close database connection."""
        if self._connection:
            self._connection.close()
            self._connection = None
    
    @contextmanager
    def transaction(self):
        """Context manager for database transactions."""
        conn = self.connect()
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Database transaction failed: {e}")
            raise
    
    def execute_query(self, query: str, params: Optional[tuple] = None) -> List[sqlite3.Row]:
        """Execute a SELECT query and return results.
        
        Args:
            query: SQL query string
            params: Query parameters
            
        Returns:
            List of query results
        """
        conn = self.connect()
        cursor = conn.cursor()
        
        try:
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            return cursor.fetchall()
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise
    
    def execute_update(self, query: str, params: Optional[tuple] = None) -> int:
        """Execute an INSERT, UPDATE, or DELETE query.
        
        Args:
            query: SQL query string
            params: Query parameters
            
        Returns:
            Number of affected rows
        """
        with self.transaction() as conn:
            cursor = conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            return cursor.rowcount
    
    def execute_many(self, query: str, params_list: List[tuple]) -> int:
        """Execute a query with multiple parameter sets.
        
        Args:
            query: SQL query string
            params_list: List of parameter tuples
            
        Returns:
            Number of affected rows
        """
        with self.transaction() as conn:
            cursor = conn.cursor()
            cursor.executemany(query, params_list)
            return cursor.rowcount
    
    def get_table_info(self, table_name: str) -> List[Dict[str, Any]]:
        """Get table schema information.
        
        Args:
            table_name: Name of the table
            
        Returns:
            List of column information dictionaries
        """
        query = f"PRAGMA table_info({table_name})"
        rows = self.execute_query(query)
        return [dict(row) for row in rows]
    
    def table_exists(self, table_name: str) -> bool:
        """Check if a table exists in the database.
        
        Args:
            table_name: Name of the table to check
            
        Returns:
            True if table exists, False otherwise
        """
        query = """
        SELECT name FROM sqlite_master 
        WHERE type='table' AND name=?
        """
        result = self.execute_query(query, (table_name,))
        return len(result) > 0
    
    def get_row_count(self, table_name: str, where_clause: str = "") -> int:
        """Get the number of rows in a table.
        
        Args:
            table_name: Name of the table
            where_clause: Optional WHERE clause (without WHERE keyword)
            
        Returns:
            Number of rows
        """
        query = f"SELECT COUNT(*) as count FROM {table_name}"
        if where_clause:
            query += f" WHERE {where_clause}"
        
        result = self.execute_query(query)
        return result[0]['count'] if result else 0
    
    def vacuum(self):
        """Optimize database by running VACUUM command."""
        conn = self.connect()
        conn.execute("VACUUM")
        logger.info("Database vacuum completed")
    
    def analyze(self):
        """Update database statistics for query optimization."""
        conn = self.connect()
        conn.execute("ANALYZE")
        logger.info("Database analyze completed")


# Global database instance
_db_instance: Optional[DatabaseConnection] = None


def get_database(db_path: str = "autonomous_demand_forecasting.db") -> DatabaseConnection:
    """Get or create global database instance.
    
    Args:
        db_path: Path to database file
        
    Returns:
        DatabaseConnection instance
    """
    global _db_instance
    if _db_instance is None or _db_instance.db_path != db_path:
        _db_instance = DatabaseConnection(db_path)
    return _db_instance


def initialize_database(db_path: str = "autonomous_demand_forecasting.db", 
                       schema_path: Optional[str] = None) -> DatabaseConnection:
    """Initialize database with schema.
    
    Args:
        db_path: Path to database file
        schema_path: Path to schema SQL file
        
    Returns:
        Initialized DatabaseConnection instance
    """
    db = get_database(db_path)
    
    # Load and execute schema
    if schema_path is None:
        schema_path = Path(__file__).parent / "schema.sql"
    
    if not os.path.exists(schema_path):
        raise FileNotFoundError(f"Schema file not found: {schema_path}")
    
    with open(schema_path, 'r') as f:
        schema_sql = f.read()
    
    # Execute schema in transaction
    with db.transaction() as conn:
        conn.executescript(schema_sql)
    
    logger.info(f"Database initialized at {db_path}")
    return db


def close_database():
    """Close global database connection."""
    global _db_instance
    if _db_instance:
        _db_instance.close()
        _db_instance = None