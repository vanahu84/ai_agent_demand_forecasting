-- Autonomous Demand Forecasting Database Schema
-- Core retail database with sales, inventory, and model lifecycle tables

-- Core retail tables
CREATE TABLE IF NOT EXISTS products (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    category TEXT NOT NULL,
    subcategory TEXT,
    brand TEXT,
    unit_price REAL,
    cost REAL,
    active BOOLEAN DEFAULT TRUE,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS stores (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    location TEXT NOT NULL,
    region TEXT,
    store_type TEXT,
    square_footage INTEGER,
    active BOOLEAN DEFAULT TRUE,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Sales data tables
CREATE TABLE IF NOT EXISTS sales_transactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    transaction_id TEXT UNIQUE NOT NULL,
    product_id TEXT NOT NULL,
    store_id TEXT NOT NULL,
    category TEXT NOT NULL,
    quantity INTEGER NOT NULL,
    unit_price REAL NOT NULL,
    total_amount REAL NOT NULL,
    customer_segment TEXT,
    transaction_date DATETIME NOT NULL,
    promotion_applied BOOLEAN DEFAULT FALSE,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (product_id) REFERENCES products(id),
    FOREIGN KEY (store_id) REFERENCES stores(id)
);

CREATE TABLE IF NOT EXISTS customer_behavior (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_segment TEXT NOT NULL,
    product_category TEXT NOT NULL,
    avg_purchase_frequency REAL,
    seasonal_multiplier REAL,
    price_sensitivity REAL,
    analysis_date DATETIME NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Inventory tables
CREATE TABLE IF NOT EXISTS inventory_levels (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    product_id TEXT NOT NULL,
    store_id TEXT NOT NULL,
    current_stock INTEGER NOT NULL,
    reserved_stock INTEGER DEFAULT 0,
    available_stock INTEGER NOT NULL,
    reorder_point INTEGER,
    max_stock_level INTEGER,
    last_updated DATETIME NOT NULL,
    FOREIGN KEY (product_id) REFERENCES products(id),
    FOREIGN KEY (store_id) REFERENCES stores(id),
    UNIQUE(product_id, store_id)
);

CREATE TABLE IF NOT EXISTS stockout_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    product_id TEXT NOT NULL,
    store_id TEXT NOT NULL,
    stockout_date DATETIME NOT NULL,
    duration_hours INTEGER,
    lost_sales_estimate REAL,
    restock_date DATETIME,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (product_id) REFERENCES products(id),
    FOREIGN KEY (store_id) REFERENCES stores(id)
);

-- Model lifecycle tables
CREATE TABLE IF NOT EXISTS model_registry (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id TEXT UNIQUE NOT NULL,
    model_name TEXT NOT NULL,
    version TEXT NOT NULL,
    model_type TEXT NOT NULL,
    status TEXT CHECK(status IN ('TRAINING', 'VALIDATION', 'PRODUCTION', 'RETIRED')) DEFAULT 'TRAINING',
    created_at DATETIME NOT NULL,
    deployed_at DATETIME,
    retired_at DATETIME,
    performance_metrics TEXT, -- JSON string
    artifact_location TEXT,
    hyperparameters TEXT, -- JSON string
    training_data_hash TEXT
);

CREATE TABLE IF NOT EXISTS model_experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id TEXT UNIQUE NOT NULL,
    model_type TEXT NOT NULL,
    hyperparameters TEXT, -- JSON string
    training_data_hash TEXT,
    training_start DATETIME NOT NULL,
    training_end DATETIME,
    accuracy_score REAL,
    mape_score REAL,
    rmse_score REAL,
    model_artifact_path TEXT,
    experiment_notes TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Model performance monitoring
CREATE TABLE IF NOT EXISTS model_performance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id TEXT NOT NULL,
    product_category TEXT NOT NULL,
    timestamp DATETIME NOT NULL,
    accuracy_score REAL,
    mape_score REAL,
    rmse_score REAL,
    prediction_count INTEGER,
    drift_score REAL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (model_id) REFERENCES model_registry(model_id)
);

CREATE TABLE IF NOT EXISTS drift_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id TEXT NOT NULL,
    severity TEXT CHECK(severity IN ('LOW', 'MEDIUM', 'HIGH')) NOT NULL,
    detected_at DATETIME NOT NULL,
    resolved_at DATETIME,
    accuracy_drop REAL NOT NULL,
    affected_categories TEXT, -- JSON array
    drift_analysis TEXT, -- JSON object
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (model_id) REFERENCES model_registry(model_id)
);

-- Validation and deployment tables
CREATE TABLE IF NOT EXISTS validation_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id TEXT NOT NULL,
    validation_dataset_id TEXT NOT NULL,
    validation_date DATETIME NOT NULL,
    accuracy_score REAL,
    baseline_accuracy REAL,
    improvement_percentage REAL,
    statistical_significance REAL,
    validation_status TEXT CHECK(validation_status IN ('PASSED', 'FAILED', 'PENDING')) DEFAULT 'PENDING',
    validation_notes TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (model_id) REFERENCES model_registry(model_id)
);

CREATE TABLE IF NOT EXISTS holdout_datasets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id TEXT UNIQUE NOT NULL,
    product_categories TEXT, -- JSON array
    date_range_start DATETIME,
    date_range_end DATETIME,
    sample_size INTEGER,
    dataset_path TEXT,
    created_at DATETIME NOT NULL
);

CREATE TABLE IF NOT EXISTS deployments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    deployment_id TEXT UNIQUE NOT NULL,
    model_id TEXT NOT NULL,
    deployment_strategy TEXT DEFAULT 'blue_green',
    status TEXT CHECK(status IN ('PENDING', 'DEPLOYING', 'ACTIVE', 'ROLLED_BACK', 'FAILED')) DEFAULT 'PENDING',
    started_at DATETIME NOT NULL,
    completed_at DATETIME,
    rollback_at DATETIME,
    performance_metrics TEXT, -- JSON object
    deployment_notes TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (model_id) REFERENCES model_registry(model_id)
);

CREATE TABLE IF NOT EXISTS production_monitoring (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    deployment_id TEXT NOT NULL,
    timestamp DATETIME NOT NULL,
    accuracy_score REAL,
    prediction_latency REAL,
    error_rate REAL,
    throughput INTEGER,
    alert_triggered BOOLEAN DEFAULT FALSE,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (deployment_id) REFERENCES deployments(deployment_id)
);

-- Retraining workflow tracking
CREATE TABLE IF NOT EXISTS retraining_workflows (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    workflow_id TEXT UNIQUE NOT NULL,
    trigger_reason TEXT NOT NULL,
    started_at DATETIME NOT NULL,
    completed_at DATETIME,
    status TEXT CHECK(status IN ('RUNNING', 'COMPLETED', 'FAILED', 'CANCELLED')) DEFAULT 'RUNNING',
    models_trained INTEGER DEFAULT 0,
    models_deployed INTEGER DEFAULT 0,
    business_impact_score REAL,
    workflow_metadata TEXT, -- JSON object
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS business_impact (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    deployment_id TEXT NOT NULL,
    metric_type TEXT NOT NULL,
    baseline_value REAL,
    improved_value REAL,
    improvement_percentage REAL,
    revenue_impact REAL,
    calculated_at DATETIME NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (deployment_id) REFERENCES deployments(deployment_id)
);

-- Indexes for performance optimization
CREATE INDEX IF NOT EXISTS idx_sales_transactions_date ON sales_transactions(transaction_date);
CREATE INDEX IF NOT EXISTS idx_sales_transactions_product ON sales_transactions(product_id);
CREATE INDEX IF NOT EXISTS idx_sales_transactions_category ON sales_transactions(category);
CREATE INDEX IF NOT EXISTS idx_model_performance_timestamp ON model_performance(timestamp);
CREATE INDEX IF NOT EXISTS idx_model_performance_model_id ON model_performance(model_id);
CREATE INDEX IF NOT EXISTS idx_drift_events_detected_at ON drift_events(detected_at);
CREATE INDEX IF NOT EXISTS idx_inventory_levels_product_store ON inventory_levels(product_id, store_id);
CREATE INDEX IF NOT EXISTS idx_deployments_status ON deployments(status);
CREATE INDEX IF NOT EXISTS idx_retraining_workflows_status ON retraining_workflows(status);