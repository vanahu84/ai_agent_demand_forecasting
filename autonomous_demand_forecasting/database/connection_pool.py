"""
Database connection pooling for autonomous demand forecasting system.

This module provides a thread-safe connection pool to prevent database
contention issues when multiple MCP servers access the database simultaneously.
"""

import sqlite3
import threading
import time
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Generator
import queue

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseConnectionPool:
    """Thread-safe SQLite connection pool."""
    
    def __init__(self, database_path: str, max_connections: int = 10, timeout: float = 30.0):
        """
        Initialize the connection pool.
        
        Args:
            database_path: Path to the SQLite database
            max_connections: Maximum number of connections in the pool
            timeout: Timeout for getting a connection from the pool
        """
        self.database_path = database_path
        self.max_connections = max_connections
        self.timeout = timeout
        self._pool = queue.Queue(maxsize=max_connections)
        self._all_connections = set()
        self._lock = threading.Lock()
        self._created_connections = 0
        
        # Ensure database directory exists
        Path(database_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Pre-create some connections
        self._initialize_pool()
        
        logger.info(f"Database connection pool initialized: {database_path} (max: {max_connections})")
    
    def _initialize_pool(self):
        """Pre-create initial connections."""
        initial_connections = min(3, self.max_connections)  # Start with 3 connections
        
        for _ in range(initial_connections):
            try:
                conn = self._create_connection()
                self._pool.put(conn, block=False)
            except Exception as e:
                logger.error(f"Failed to create initial connection: {e}")
    
    def _create_connection(self) -> sqlite3.Connection:
        """Create a new database connection with optimized settings."""
        conn = sqlite3.connect(
            self.database_path,
            timeout=20.0,  # 20 second timeout for database operations
            check_same_thread=False,  # Allow connection sharing across threads
            isolation_level=None  # Autocommit mode for better concurrency
        )
        
        # Optimize SQLite settings for concurrent access
        conn.execute("PRAGMA journal_mode=WAL")  # Write-Ahead Logging for better concurrency
        conn.execute("PRAGMA synchronous=NORMAL")  # Balance between safety and performance
        conn.execute("PRAGMA cache_size=10000")  # Larger cache for better performance
        conn.execute("PRAGMA temp_store=MEMORY")  # Store temp tables in memory
        conn.execute("PRAGMA mmap_size=268435456")  # 256MB memory-mapped I/O
        
        # Set row factory for easier data access
        conn.row_factory = sqlite3.Row
        
        with self._lock:
            self._all_connections.add(conn)
            self._created_connections += 1
        
        logger.debug(f"Created new database connection (total: {self._created_connections})")
        return conn
    
    def get_connection(self) -> sqlite3.Connection:
        """
        Get a connection from the pool.
        
        Returns:
            A database connection
            
        Raises:
            TimeoutError: If no connection is available within the timeout period
        """
        try:
            # Try to get an existing connection from the pool
            conn = self._pool.get(timeout=self.timeout)
            
            # Test the connection to make sure it's still valid
            try:
                conn.execute("SELECT 1").fetchone()
                return conn
            except sqlite3.Error:
                # Connection is stale, create a new one
                logger.warning("Stale connection detected, creating new one")
                with self._lock:
                    self._all_connections.discard(conn)
                try:
                    conn.close()
                except:
                    pass
                
        except queue.Empty:
            # No connection available in pool
            pass
        
        # Create a new connection if we haven't reached the limit
        with self._lock:
            if self._created_connections < self.max_connections:
                return self._create_connection()
        
        # Pool is full, wait for a connection to be returned
        try:
            conn = self._pool.get(timeout=self.timeout)
            # Test the connection
            conn.execute("SELECT 1").fetchone()
            return conn
        except queue.Empty:
            raise TimeoutError(f"Could not get database connection within {self.timeout} seconds")
        except sqlite3.Error:
            # Connection is stale, try one more time
            with self._lock:
                if self._created_connections < self.max_connections:
                    return self._create_connection()
            raise TimeoutError("Could not get a valid database connection")
    
    def return_connection(self, conn: sqlite3.Connection):
        """
        Return a connection to the pool.
        
        Args:
            conn: The connection to return
        """
        if conn in self._all_connections:
            try:
                # Test the connection before returning it
                conn.execute("SELECT 1").fetchone()
                self._pool.put(conn, block=False)
            except (sqlite3.Error, queue.Full):
                # Connection is bad or pool is full, close it
                with self._lock:
                    self._all_connections.discard(conn)
                    self._created_connections -= 1
                try:
                    conn.close()
                except:
                    pass
                logger.debug("Closed bad or excess connection")
    
    @contextmanager
    def get_connection_context(self) -> Generator[sqlite3.Connection, None, None]:
        """
        Context manager for getting and automatically returning connections.
        
        Usage:
            with pool.get_connection_context() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM table")
                result = cursor.fetchall()
        """
        conn = self.get_connection()
        try:
            yield conn
        finally:
            self.return_connection(conn)
    
    def close_all(self):
        """Close all connections in the pool."""
        with self._lock:
            # Close all connections in the pool
            while not self._pool.empty():
                try:
                    conn = self._pool.get_nowait()
                    conn.close()
                except (queue.Empty, sqlite3.Error):
                    pass
            
            # Close any remaining connections
            for conn in list(self._all_connections):
                try:
                    conn.close()
                except sqlite3.Error:
                    pass
            
            self._all_connections.clear()
            self._created_connections = 0
        
        logger.info("All database connections closed")
    
    def get_stats(self) -> dict:
        """Get connection pool statistics."""
        with self._lock:
            return {
                "total_connections": self._created_connections,
                "available_connections": self._pool.qsize(),
                "max_connections": self.max_connections,
                "active_connections": self._created_connections - self._pool.qsize()
            }


# Global connection pool instance
_connection_pool: Optional[DatabaseConnectionPool] = None
_pool_lock = threading.Lock()

def get_connection_pool() -> DatabaseConnectionPool:
    """Get the global connection pool instance."""
    global _connection_pool
    
    if _connection_pool is None:
        with _pool_lock:
            if _connection_pool is None:
                database_path = Path(__file__).parent / "autonomous_demand_forecasting.db"
                _connection_pool = DatabaseConnectionPool(
                    database_path=str(database_path),
                    max_connections=15,  # Allow more connections for multiple MCP servers
                    timeout=10.0  # Reasonable timeout
                )
    
    return _connection_pool

def get_db_connection() -> sqlite3.Connection:
    """
    Get a database connection from the pool.
    
    This is a drop-in replacement for the old get_db_connection() function
    used throughout the MCP servers.
    """
    return get_connection_pool().get_connection()

@contextmanager
def get_db_connection_context() -> Generator[sqlite3.Connection, None, None]:
    """
    Context manager for database connections.
    
    Usage:
        with get_db_connection_context() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM table")
            result = cursor.fetchall()
    """
    pool = get_connection_pool()
    with pool.get_connection_context() as conn:
        yield conn

def return_db_connection(conn: sqlite3.Connection):
    """Return a connection to the pool."""
    get_connection_pool().return_connection(conn)

def close_connection_pool():
    """Close the global connection pool."""
    global _connection_pool
    if _connection_pool:
        _connection_pool.close_all()
        _connection_pool = None

def get_pool_stats() -> dict:
    """Get connection pool statistics."""
    return get_connection_pool().get_stats()


# Cleanup function for graceful shutdown
import atexit
atexit.register(close_connection_pool)