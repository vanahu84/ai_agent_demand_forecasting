"""
Database utility functions for retail-specific operations.
"""

import json
import sqlite3
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import logging

from .connection import get_database
from .models import (
    Product, Store, SalesTransaction, CustomerBehavior, InventoryLevel,
    StockoutEvent, ModelRegistry, ModelExperiment, AccuracyMetrics,
    DriftEvent, ValidationResult, HoldoutDataset, DeploymentResult,
    ProductionMonitoring, RetrainingWorkflow, BusinessImpact,
    SeverityLevel, ModelStatus, ValidationStatus, DeploymentStatus,
    WorkflowStatus
)

logger = logging.getLogger(__name__)


class RetailDatabaseUtils:
    """Utility functions for retail database operations."""
    
    def __init__(self, db_path: str = "autonomous_demand_forecasting.db"):
        """Initialize with database connection.
        
        Args:
            db_path: Path to database file
        """
        self.db = get_database(db_path)
    
    # Product operations
    def insert_product(self, product: Product) -> str:
        """Insert a new product.
        
        Args:
            product: Product data model
            
        Returns:
            Product ID
        """
        query = """
        INSERT INTO products (id, name, category, subcategory, brand, unit_price, cost, active)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            product.id, product.name, product.category, product.subcategory,
            product.brand, product.unit_price, product.cost, product.active
        )
        self.db.execute_update(query, params)
        return product.id
    
    def get_product(self, product_id: str) -> Optional[Product]:
        """Get product by ID.
        
        Args:
            product_id: Product identifier
            
        Returns:
            Product data model or None
        """
        query = "SELECT * FROM products WHERE id = ?"
        result = self.db.execute_query(query, (product_id,))
        
        if result:
            row = result[0]
            return Product(
                id=row['id'],
                name=row['name'],
                category=row['category'],
                subcategory=row['subcategory'],
                brand=row['brand'],
                unit_price=row['unit_price'],
                cost=row['cost'],
                active=bool(row['active']),
                created_at=datetime.fromisoformat(row['created_at']) if row['created_at'] else None,
                updated_at=datetime.fromisoformat(row['updated_at']) if row['updated_at'] else None
            )
        return None
    
    def get_products_by_category(self, category: str) -> List[Product]:
        """Get all products in a category.
        
        Args:
            category: Product category
            
        Returns:
            List of Product data models
        """
        query = "SELECT * FROM products WHERE category = ? AND active = 1"
        results = self.db.execute_query(query, (category,))
        
        products = []
        for row in results:
            products.append(Product(
                id=row['id'],
                name=row['name'],
                category=row['category'],
                subcategory=row['subcategory'],
                brand=row['brand'],
                unit_price=row['unit_price'],
                cost=row['cost'],
                active=bool(row['active']),
                created_at=datetime.fromisoformat(row['created_at']) if row['created_at'] else None,
                updated_at=datetime.fromisoformat(row['updated_at']) if row['updated_at'] else None
            ))
        return products
    
    # Sales transaction operations
    def insert_sales_transaction(self, transaction: SalesTransaction) -> int:
        """Insert a sales transaction.
        
        Args:
            transaction: SalesTransaction data model
            
        Returns:
            Transaction record ID
        """
        query = """
        INSERT INTO sales_transactions 
        (transaction_id, product_id, store_id, category, quantity, unit_price, 
         total_amount, customer_segment, transaction_date, promotion_applied)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            transaction.transaction_id, transaction.product_id, transaction.store_id,
            transaction.category, transaction.quantity, transaction.unit_price,
            transaction.total_amount, transaction.customer_segment,
            transaction.transaction_date, transaction.promotion_applied
        )
        self.db.execute_update(query, params)
        
        # Get the inserted record ID
        result = self.db.execute_query(
            "SELECT id FROM sales_transactions WHERE transaction_id = ?",
            (transaction.transaction_id,)
        )
        return result[0]['id'] if result else 0
    
    def get_sales_data(self, days_back: int = 90, category: Optional[str] = None) -> List[SalesTransaction]:
        """Get sales data for the specified period.
        
        Args:
            days_back: Number of days to look back
            category: Optional product category filter
            
        Returns:
            List of SalesTransaction data models
        """
        start_date = datetime.now() - timedelta(days=days_back)
        
        query = """
        SELECT * FROM sales_transactions 
        WHERE transaction_date >= ?
        """
        params = [start_date]
        
        if category:
            query += " AND category = ?"
            params.append(category)
        
        query += " ORDER BY transaction_date DESC"
        
        results = self.db.execute_query(query, tuple(params))
        
        transactions = []
        for row in results:
            transactions.append(SalesTransaction(
                id=row['id'],
                transaction_id=row['transaction_id'],
                product_id=row['product_id'],
                store_id=row['store_id'],
                category=row['category'],
                quantity=row['quantity'],
                unit_price=row['unit_price'],
                total_amount=row['total_amount'],
                customer_segment=row['customer_segment'],
                transaction_date=datetime.fromisoformat(row['transaction_date']),
                promotion_applied=bool(row['promotion_applied']),
                created_at=datetime.fromisoformat(row['created_at']) if row['created_at'] else None
            ))
        return transactions
    
    def get_sales_summary_by_category(self, days_back: int = 30) -> Dict[str, Dict[str, float]]:
        """Get sales summary grouped by category.
        
        Args:
            days_back: Number of days to analyze
            
        Returns:
            Dictionary with category summaries
        """
        start_date = datetime.now() - timedelta(days=days_back)
        
        query = """
        SELECT category,
               COUNT(*) as transaction_count,
               SUM(quantity) as total_quantity,
               SUM(total_amount) as total_revenue,
               AVG(total_amount) as avg_transaction_value
        FROM sales_transactions 
        WHERE transaction_date >= ?
        GROUP BY category
        ORDER BY total_revenue DESC
        """
        
        results = self.db.execute_query(query, (start_date,))
        
        summary = {}
        for row in results:
            summary[row['category']] = {
                'transaction_count': row['transaction_count'],
                'total_quantity': row['total_quantity'],
                'total_revenue': row['total_revenue'],
                'avg_transaction_value': row['avg_transaction_value']
            }
        return summary
    
    # Inventory operations
    def update_inventory_level(self, inventory: InventoryLevel) -> bool:
        """Update inventory level for a product at a store.
        
        Args:
            inventory: InventoryLevel data model
            
        Returns:
            True if successful
        """
        query = """
        INSERT OR REPLACE INTO inventory_levels 
        (product_id, store_id, current_stock, reserved_stock, available_stock,
         reorder_point, max_stock_level, last_updated)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            inventory.product_id, inventory.store_id, inventory.current_stock,
            inventory.reserved_stock, inventory.available_stock,
            inventory.reorder_point, inventory.max_stock_level, inventory.last_updated
        )
        
        rows_affected = self.db.execute_update(query, params)
        return rows_affected > 0
    
    def get_low_stock_products(self, store_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get products with low stock levels.
        
        Args:
            store_id: Optional store filter
            
        Returns:
            List of low stock product information
        """
        query = """
        SELECT il.product_id, il.store_id, il.current_stock, il.reorder_point,
               p.name, p.category
        FROM inventory_levels il
        JOIN products p ON il.product_id = p.id
        WHERE il.current_stock <= il.reorder_point
        """
        params = []
        
        if store_id:
            query += " AND il.store_id = ?"
            params.append(store_id)
        
        query += " ORDER BY il.current_stock ASC"
        
        results = self.db.execute_query(query, tuple(params))
        return [dict(row) for row in results]
    
    def record_stockout_event(self, stockout: StockoutEvent) -> int:
        """Record a stockout event.
        
        Args:
            stockout: StockoutEvent data model
            
        Returns:
            Stockout event ID
        """
        query = """
        INSERT INTO stockout_events 
        (product_id, store_id, stockout_date, duration_hours, lost_sales_estimate, restock_date)
        VALUES (?, ?, ?, ?, ?, ?)
        """
        params = (
            stockout.product_id, stockout.store_id, stockout.stockout_date,
            stockout.duration_hours, stockout.lost_sales_estimate, stockout.restock_date
        )
        self.db.execute_update(query, params)
        
        # Get the inserted record ID
        result = self.db.execute_query(
            "SELECT last_insert_rowid() as id"
        )
        return result[0]['id'] if result else 0
    
    # Model lifecycle operations
    def register_model(self, model: ModelRegistry) -> str:
        """Register a new model in the registry.
        
        Args:
            model: ModelRegistry data model
            
        Returns:
            Model ID
        """
        query = """
        INSERT INTO model_registry 
        (model_id, model_name, version, model_type, status, created_at,
         performance_metrics, artifact_location, hyperparameters, training_data_hash)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            model.model_id, model.model_name, model.version, model.model_type,
            model.status.value, model.created_at,
            json.dumps(model.performance_metrics) if model.performance_metrics else None,
            model.artifact_location,
            json.dumps(model.hyperparameters) if model.hyperparameters else None,
            model.training_data_hash
        )
        self.db.execute_update(query, params)
        return model.model_id
    
    def update_model_status(self, model_id: str, status: ModelStatus, 
                           deployed_at: Optional[datetime] = None) -> bool:
        """Update model status.
        
        Args:
            model_id: Model identifier
            status: New model status
            deployed_at: Optional deployment timestamp
            
        Returns:
            True if successful
        """
        query = "UPDATE model_registry SET status = ?"
        params = [status.value]
        
        if deployed_at:
            query += ", deployed_at = ?"
            params.append(deployed_at)
        
        query += " WHERE model_id = ?"
        params.append(model_id)
        
        rows_affected = self.db.execute_update(query, tuple(params))
        return rows_affected > 0
    
    def get_production_models(self) -> List[ModelRegistry]:
        """Get all models currently in production.
        
        Returns:
            List of ModelRegistry data models
        """
        query = "SELECT * FROM model_registry WHERE status = 'PRODUCTION'"
        results = self.db.execute_query(query)
        
        models = []
        for row in results:
            models.append(ModelRegistry(
                id=row['id'],
                model_id=row['model_id'],
                model_name=row['model_name'],
                version=row['version'],
                model_type=row['model_type'],
                status=ModelStatus(row['status']),
                created_at=datetime.fromisoformat(row['created_at']),
                deployed_at=datetime.fromisoformat(row['deployed_at']) if row['deployed_at'] else None,
                retired_at=datetime.fromisoformat(row['retired_at']) if row['retired_at'] else None,
                performance_metrics=json.loads(row['performance_metrics']) if row['performance_metrics'] else None,
                artifact_location=row['artifact_location'],
                hyperparameters=json.loads(row['hyperparameters']) if row['hyperparameters'] else None,
                training_data_hash=row['training_data_hash']
            ))
        return models
    
    # Model performance tracking
    def record_model_performance(self, model_id: str, category: str, metrics: AccuracyMetrics) -> int:
        """Record model performance metrics.
        
        Args:
            model_id: Model identifier
            category: Product category
            metrics: AccuracyMetrics data model
            
        Returns:
            Performance record ID
        """
        query = """
        INSERT INTO model_performance 
        (model_id, product_category, timestamp, accuracy_score, mape_score, 
         rmse_score, prediction_count, drift_score)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            model_id, category, metrics.timestamp, metrics.accuracy_score,
            metrics.mape_score, metrics.rmse_score, metrics.prediction_count,
            getattr(metrics, 'drift_score', None)
        )
        self.db.execute_update(query, params)
        
        # Get the inserted record ID
        result = self.db.execute_query("SELECT last_insert_rowid() as id")
        return result[0]['id'] if result else 0
    
    def get_model_performance_history(self, model_id: str, days_back: int = 30) -> List[Dict[str, Any]]:
        """Get model performance history.
        
        Args:
            model_id: Model identifier
            days_back: Number of days to look back
            
        Returns:
            List of performance records
        """
        start_date = datetime.now() - timedelta(days=days_back)
        
        query = """
        SELECT * FROM model_performance 
        WHERE model_id = ? AND timestamp >= ?
        ORDER BY timestamp DESC
        """
        
        results = self.db.execute_query(query, (model_id, start_date))
        return [dict(row) for row in results]
    
    def record_drift_event(self, drift_event: DriftEvent) -> int:
        """Record a model drift event.
        
        Args:
            drift_event: DriftEvent data model
            
        Returns:
            Drift event ID
        """
        query = """
        INSERT INTO drift_events 
        (model_id, severity, detected_at, accuracy_drop, affected_categories, drift_analysis)
        VALUES (?, ?, ?, ?, ?, ?)
        """
        params = (
            drift_event.model_id, drift_event.severity.value, drift_event.detected_at,
            drift_event.accuracy_drop, json.dumps(drift_event.affected_categories),
            json.dumps(drift_event.drift_analysis) if drift_event.drift_analysis else None
        )
        self.db.execute_update(query, params)
        
        # Get the inserted record ID
        result = self.db.execute_query("SELECT last_insert_rowid() as id")
        return result[0]['id'] if result else 0
    
    def get_active_drift_events(self, model_id: Optional[str] = None) -> List[DriftEvent]:
        """Get active (unresolved) drift events.
        
        Args:
            model_id: Optional model filter
            
        Returns:
            List of DriftEvent data models
        """
        query = "SELECT * FROM drift_events WHERE resolved_at IS NULL"
        params = []
        
        if model_id:
            query += " AND model_id = ?"
            params.append(model_id)
        
        query += " ORDER BY detected_at DESC"
        
        results = self.db.execute_query(query, tuple(params))
        
        events = []
        for row in results:
            events.append(DriftEvent(
                id=row['id'],
                model_id=row['model_id'],
                severity=SeverityLevel(row['severity']),
                detected_at=datetime.fromisoformat(row['detected_at']),
                accuracy_drop=row['accuracy_drop'],
                affected_categories=json.loads(row['affected_categories']) if row['affected_categories'] else [],
                drift_analysis=json.loads(row['drift_analysis']) if row['drift_analysis'] else None,
                resolved_at=datetime.fromisoformat(row['resolved_at']) if row['resolved_at'] else None,
                created_at=datetime.fromisoformat(row['created_at']) if row['created_at'] else None
            ))
        return events
    
    # Utility functions
    def get_database_stats(self) -> Dict[str, int]:
        """Get database statistics.
        
        Returns:
            Dictionary with table row counts
        """
        tables = [
            'products', 'stores', 'sales_transactions', 'customer_behavior',
            'inventory_levels', 'stockout_events', 'model_registry',
            'model_experiments', 'model_performance', 'drift_events',
            'validation_results', 'deployments', 'retraining_workflows'
        ]
        
        stats = {}
        for table in tables:
            stats[table] = self.db.get_row_count(table)
        
        return stats
    
    def cleanup_old_records(self, days_to_keep: int = 365):
        """Clean up old records to maintain database performance.
        
        Args:
            days_to_keep: Number of days of data to retain
        """
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        # Clean up old performance records
        query = "DELETE FROM model_performance WHERE timestamp < ?"
        deleted = self.db.execute_update(query, (cutoff_date,))
        logger.info(f"Cleaned up {deleted} old performance records")
        
        # Clean up old sales transactions (keep more recent data)
        sales_cutoff = datetime.now() - timedelta(days=days_to_keep * 2)
        query = "DELETE FROM sales_transactions WHERE transaction_date < ?"
        deleted = self.db.execute_update(query, (sales_cutoff,))
        logger.info(f"Cleaned up {deleted} old sales transactions")
        
        # Vacuum database after cleanup
        self.db.vacuum()


# Global utility instance
_utils_instance: Optional[RetailDatabaseUtils] = None


def get_retail_db_utils(db_path: str = "autonomous_demand_forecasting.db") -> RetailDatabaseUtils:
    """Get or create global retail database utils instance.
    
    Args:
        db_path: Path to database file
        
    Returns:
        RetailDatabaseUtils instance
    """
    global _utils_instance
    if _utils_instance is None:
        _utils_instance = RetailDatabaseUtils(db_path)
    return _utils_instance