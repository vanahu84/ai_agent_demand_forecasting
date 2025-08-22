#!/usr/bin/env python3
"""
Database initialization script for autonomous demand forecasting system.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from autonomous_demand_forecasting.database.migrations import (
    setup_database, verify_database_integrity
)


def initialize_database(db_path: str, include_sample_data: bool = False) -> bool:
    """Initialize database for testing purposes."""
    try:
        return setup_database(db_path, include_sample_data)
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        return False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main initialization function."""
    parser = argparse.ArgumentParser(
        description="Initialize autonomous demand forecasting database"
    )
    parser.add_argument(
        "--db-path",
        default="autonomous_demand_forecasting.db",
        help="Path to database file (default: autonomous_demand_forecasting.db)"
    )
    parser.add_argument(
        "--sample-data",
        action="store_true",
        help="Include sample data for testing"
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing database integrity"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recreation of existing database"
    )
    
    args = parser.parse_args()
    
    try:
        if args.verify_only:
            # Verify existing database
            logger.info("Verifying database integrity...")
            result = verify_database_integrity(args.db_path)
            
            print("\n=== Database Integrity Report ===")
            print(f"Status: {result['status']}")
            print(f"Database Path: {result.get('database_path', 'N/A')}")
            
            if result['status'] == 'error':
                print(f"Error: {result['message']}")
                return 1
            
            print(f"File Size: {result.get('file_size_mb', 0)} MB")
            print(f"Integrity Check: {result.get('integrity_check', 'unknown')}")
            
            if result.get('missing_tables'):
                print(f"Missing Tables: {', '.join(result['missing_tables'])}")
            
            print("\nTable Statistics:")
            for table, count in result.get('table_stats', {}).items():
                print(f"  {table}: {count} records")
            
            return 0 if result['status'] == 'ok' else 1
        
        else:
            # Setup database
            if Path(args.db_path).exists() and not args.force:
                logger.warning(f"Database {args.db_path} already exists. Use --force to recreate.")
                return 1
            
            if args.force and Path(args.db_path).exists():
                logger.info(f"Removing existing database {args.db_path}")
                Path(args.db_path).unlink()
            
            logger.info(f"Setting up database at {args.db_path}")
            success = setup_database(args.db_path, args.sample_data)
            
            if success:
                logger.info("Database setup completed successfully")
                
                # Verify the setup
                result = verify_database_integrity(args.db_path)
                print(f"\nDatabase created with {sum(result.get('table_stats', {}).values())} total records")
                
                return 0
            else:
                logger.error("Database setup failed")
                return 1
    
    except KeyboardInterrupt:
        logger.info("Setup interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())