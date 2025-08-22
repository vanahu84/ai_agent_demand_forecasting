# Autonomous Demand Forecasting Database

This module provides the core database schema, data models, and utilities for the autonomous demand forecasting system.

## Components

### 1. Schema (`schema.sql`)
Complete SQLite database schema including:
- **Retail tables**: products, stores, sales_transactions, customer_behavior
- **Inventory tables**: inventory_levels, stockout_events  
- **Model lifecycle tables**: model_registry, model_experiments, model_performance
- **Monitoring tables**: drift_events, validation_results, deployments
- **Workflow tables**: retraining_workflows, business_impact

### 2. Data Models (`models.py`)
Python dataclasses for all database entities:
- Core retail models: `Product`, `Store`, `SalesTransaction`, `InventoryLevel`
- ML models: `ModelRegistry`, `ModelExperiment`, `AccuracyMetrics`
- Monitoring models: `DriftEvent`, `ValidationResult`, `DeploymentResult`
- Workflow models: `RetrainingWorkflow`, `BusinessImpact`

### 3. Database Connection (`connection.py`)
- `DatabaseConnection`: SQLite connection manager with optimizations
- Connection pooling and transaction management
- Performance optimizations (WAL mode, caching)

### 4. Utilities (`utils.py`)
- `RetailDatabaseUtils`: High-level operations for retail data
- Product and inventory management functions
- Sales data collection and analysis
- Model performance tracking
- Drift event recording and retrieval

### 5. Migrations (`migrations.py`)
- Database initialization and migration management
- Sample data population for testing
- Database integrity verification
- Schema versioning and updates

## Quick Start

### Initialize Database
```python
from autonomous_demand_forecasting.database.migrations import setup_database

# Create database with schema
setup_database("my_database.db", include_sample_data=True)
```

### Use Database Utilities
```python
from autonomous_demand_forecasting.database.utils import get_retail_db_utils
from autonomous_demand_forecasting.database.models import Product

utils = get_retail_db_utils("my_database.db")

# Add a product
product = Product(
    id="PROD001",
    name="Wireless Headphones", 
    category="Electronics",
    unit_price=99.99
)
utils.insert_product(product)

# Get sales data
sales_data = utils.get_sales_data(days_back=30)
```

### Command Line Setup
```bash
# Initialize database
python autonomous_demand_forecasting/database/init_db.py --sample-data

# Verify database
python autonomous_demand_forecasting/database/init_db.py --verify-only
```

## Database Schema Overview

### Core Tables
- `products`: Product catalog with pricing and categorization
- `stores`: Store locations and metadata
- `sales_transactions`: Individual sales records
- `inventory_levels`: Current stock levels by product/store
- `customer_behavior`: Aggregated customer analytics

### Model Lifecycle Tables  
- `model_registry`: Registered ML models with metadata
- `model_experiments`: Training experiment results
- `model_performance`: Real-time accuracy tracking
- `drift_events`: Model degradation detection
- `validation_results`: Model validation outcomes
- `deployments`: Production deployment tracking

### Workflow Tables
- `retraining_workflows`: Autonomous retraining processes
- `business_impact`: ROI and business metrics

## Performance Features

- **Optimized indexes** for common query patterns
- **WAL mode** for concurrent read/write operations  
- **Connection pooling** for efficient resource usage
- **Batch operations** for high-volume data processing
- **Automatic cleanup** of old records

## Requirements Satisfied

This implementation satisfies the following requirements from the specification:

- **Requirement 2.3**: Data quality validation and cleaning capabilities
- **Requirement 3.3**: Model artifact generation and registry management  
- **Requirement 4.2**: Statistical validation and comparison algorithms
- **Requirement 5.1**: Model deployment and artifact management
- **Requirement 9.1**: MLOps pipeline and automation infrastructure

## Testing

Run the test suite to verify functionality:
```bash
python test_database_setup.py
```

The test suite validates:
- Database schema creation
- Data model functionality
- CRUD operations
- Sample data population
- Integrity checks