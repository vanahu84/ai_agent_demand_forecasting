"""
Database migration scripts and initialization procedures.
"""

import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

from .connection import get_database, initialize_database
from .models import Product, Store

logger = logging.getLogger(__name__)


class DatabaseMigration:
    """Database migration management."""
    
    def __init__(self, db_path: str = "autonomous_demand_forecasting.db"):
        """Initialize migration manager.
        
        Args:
            db_path: Path to database file
        """
        self.db_path = db_path
        self.db = get_database(db_path)
        self._ensure_migration_table()
    
    def _ensure_migration_table(self):
        """Ensure migration tracking table exists."""
        query = """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            version TEXT UNIQUE NOT NULL,
            description TEXT,
            applied_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            checksum TEXT
        )
        """
        self.db.execute_update(query)
    
    def get_applied_migrations(self) -> List[Dict[str, Any]]:
        """Get list of applied migrations.
        
        Returns:
            List of migration records
        """
        query = "SELECT * FROM schema_migrations ORDER BY applied_at"
        results = self.db.execute_query(query)
        return [dict(row) for row in results]
    
    def is_migration_applied(self, version: str) -> bool:
        """Check if a migration version has been applied.
        
        Args:
            version: Migration version string
            
        Returns:
            True if migration is applied
        """
        query = "SELECT COUNT(*) as count FROM schema_migrations WHERE version = ?"
        result = self.db.execute_query(query, (version,))
        return result[0]['count'] > 0 if result else False
    
    def apply_migration(self, version: str, description: str, sql_commands: List[str]):
        """Apply a database migration.
        
        Args:
            version: Migration version string
            description: Migration description
            sql_commands: List of SQL commands to execute
        """
        if self.is_migration_applied(version):
            logger.info(f"Migration {version} already applied, skipping")
            return
        
        logger.info(f"Applying migration {version}: {description}")
        
        try:
            with self.db.transaction() as conn:
                # Execute migration commands
                for command in sql_commands:
                    if command.strip():
                        conn.execute(command)
                
                # Record migration
                conn.execute(
                    "INSERT INTO schema_migrations (version, description) VALUES (?, ?)",
                    (version, description)
                )
            
            logger.info(f"Migration {version} applied successfully")
            
        except Exception as e:
            logger.error(f"Migration {version} failed: {e}")
            raise
    
    def rollback_migration(self, version: str):
        """Rollback a migration (manual process).
        
        Args:
            version: Migration version to rollback
        """
        logger.warning(f"Manual rollback required for migration {version}")
        # Note: SQLite doesn't support easy rollbacks, this would need manual intervention


def create_initial_schema(db_path: str = "autonomous_demand_forecasting.db") -> bool:
    """Create initial database schema.
    
    Args:
        db_path: Path to database file
        
    Returns:
        True if successful
    """
    try:
        db = initialize_database(db_path)
        logger.info("Initial schema created successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to create initial schema: {e}")
        return False


def populate_sample_data(db_path: str = "autonomous_demand_forecasting.db") -> bool:
    """Populate database with sample retail data for testing.
    
    Args:
        db_path: Path to database file
        
    Returns:
        True if successful
    """
    try:
        from .utils import get_retail_db_utils
        
        utils = get_retail_db_utils(db_path)
        
        # Sample products
        sample_products = [
            Product(
                id="PROD001",
                name="Wireless Headphones",
                category="Electronics",
                subcategory="Audio",
                brand="TechBrand",
                unit_price=99.99,
                cost=45.00
            ),
            Product(
                id="PROD002",
                name="Running Shoes",
                category="Footwear",
                subcategory="Athletic",
                brand="SportsBrand",
                unit_price=129.99,
                cost=65.00
            ),
            Product(
                id="PROD003",
                name="Coffee Maker",
                category="Appliances",
                subcategory="Kitchen",
                brand="HomeBrand",
                unit_price=79.99,
                cost=40.00
            ),
            Product(
                id="PROD004",
                name="Yoga Mat",
                category="Fitness",
                subcategory="Equipment",
                brand="FitBrand",
                unit_price=29.99,
                cost=12.00
            ),
            Product(
                id="PROD005",
                name="Smartphone Case",
                category="Electronics",
                subcategory="Accessories",
                brand="TechBrand",
                unit_price=19.99,
                cost=8.00
            )
        ]
        
        # Sample stores
        sample_stores = [
            Store(
                id="STORE001",
                name="Downtown Main",
                location="123 Main St, Downtown",
                region="Central",
                store_type="Flagship",
                square_footage=5000
            ),
            Store(
                id="STORE002",
                name="Mall Location",
                location="456 Mall Ave, Shopping Center",
                region="North",
                store_type="Standard",
                square_footage=3000
            ),
            Store(
                id="STORE003",
                name="Outlet Store",
                location="789 Outlet Rd, Industrial",
                region="South",
                store_type="Outlet",
                square_footage=2000
            )
        ]
        
        # Insert sample data
        for product in sample_products:
            utils.insert_product(product)
        
        # Insert stores directly (utils doesn't have store insert method yet)
        for store in sample_stores:
            query = """
            INSERT INTO stores (id, name, location, region, store_type, square_footage)
            VALUES (?, ?, ?, ?, ?, ?)
            """
            params = (store.id, store.name, store.location, store.region, 
                     store.store_type, store.square_footage)
            utils.db.execute_update(query, params)
        
        logger.info("Sample data populated successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to populate sample data: {e}")
        return False


def run_all_migrations(db_path: str = "autonomous_demand_forecasting.db") -> bool:
    """Run all available migrations.
    
    Args:
        db_path: Path to database file
        
    Returns:
        True if all migrations successful
    """
    migration_manager = DatabaseMigration(db_path)
    
    # Define migrations
    migrations = [
        {
            "version": "001_initial_schema",
            "description": "Create initial database schema",
            "commands": []  # Schema is created by initialize_database
        },
        {
            "version": "002_add_indexes",
            "description": "Add performance indexes",
            "commands": [
                "CREATE INDEX IF NOT EXISTS idx_sales_date_category ON sales_transactions(transaction_date, category)",
                "CREATE INDEX IF NOT EXISTS idx_model_perf_model_timestamp ON model_performance(model_id, timestamp)",
                "CREATE INDEX IF NOT EXISTS idx_inventory_product_store ON inventory_levels(product_id, store_id)",
                "CREATE INDEX IF NOT EXISTS idx_drift_events_model_severity ON drift_events(model_id, severity)"
            ]
        },
        {
            "version": "003_add_triggers",
            "description": "Add database triggers for data integrity",
            "commands": [
                """
                CREATE TRIGGER IF NOT EXISTS update_product_timestamp 
                AFTER UPDATE ON products
                BEGIN
                    UPDATE products SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
                END
                """,
                """
                CREATE TRIGGER IF NOT EXISTS validate_inventory_stock
                BEFORE INSERT ON inventory_levels
                BEGIN
                    SELECT CASE
                        WHEN NEW.current_stock < 0 THEN
                            RAISE(ABORT, 'Current stock cannot be negative')
                        WHEN NEW.available_stock > NEW.current_stock THEN
                            RAISE(ABORT, 'Available stock cannot exceed current stock')
                    END;
                END
                """
            ]
        }
    ]
    
    try:
        # Ensure database exists with initial schema
        if not os.path.exists(db_path):
            create_initial_schema(db_path)
        
        # Apply each migration
        for migration in migrations:
            if migration["commands"]:  # Skip empty command lists
                migration_manager.apply_migration(
                    migration["version"],
                    migration["description"],
                    migration["commands"]
                )
        
        logger.info("All migrations completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        return False


def setup_database(db_path: str = "autonomous_demand_forecasting.db", 
                  include_sample_data: bool = False) -> bool:
    """Complete database setup with schema and optional sample data.
    
    Args:
        db_path: Path to database file
        include_sample_data: Whether to include sample data
        
    Returns:
        True if setup successful
    """
    try:
        logger.info(f"Setting up database at {db_path}")
        
        # Create initial schema
        if not create_initial_schema(db_path):
            return False
        
        # Run migrations
        if not run_all_migrations(db_path):
            return False
        
        # Add sample data if requested
        if include_sample_data:
            if not populate_sample_data(db_path):
                logger.warning("Sample data population failed, but continuing")
        
        logger.info("Database setup completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Database setup failed: {e}")
        return False


def verify_database_integrity(db_path: str = "autonomous_demand_forecasting.db") -> Dict[str, Any]:
    """Verify database integrity and return status report.
    
    Args:
        db_path: Path to database file
        
    Returns:
        Dictionary with integrity check results
    """
    try:
        from .utils import get_retail_db_utils
        
        utils = get_retail_db_utils(db_path)
        
        # Check if database file exists
        if not os.path.exists(db_path):
            return {"status": "error", "message": "Database file does not exist"}
        
        # Get database statistics
        stats = utils.get_database_stats()
        
        # Check for required tables
        required_tables = [
            'products', 'stores', 'sales_transactions', 'inventory_levels',
            'model_registry', 'model_performance', 'drift_events'
        ]
        
        missing_tables = []
        for table in required_tables:
            if not utils.db.table_exists(table):
                missing_tables.append(table)
        
        # Run integrity check
        integrity_results = utils.db.execute_query("PRAGMA integrity_check")
        integrity_ok = len(integrity_results) == 1 and integrity_results[0][0] == "ok"
        
        return {
            "status": "ok" if not missing_tables and integrity_ok else "warning",
            "database_path": db_path,
            "file_exists": True,
            "file_size_mb": round(os.path.getsize(db_path) / (1024 * 1024), 2),
            "table_stats": stats,
            "missing_tables": missing_tables,
            "integrity_check": "passed" if integrity_ok else "failed",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }