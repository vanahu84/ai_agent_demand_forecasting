#!/usr/bin/env python3
"""
Demo script for the Monitoring Dashboard
Shows the dashboard with sample data
"""

import sqlite3
import os
from datetime import datetime, timedelta
from monitoring_dashboard import MonitoringDashboard

def setup_demo_database(db_path: str):
    """Set up demo database with sample data"""
    
    # Remove existing database if it exists
    if os.path.exists(db_path):
        os.remove(db_path)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create tables
    cursor.executescript("""
        CREATE TABLE model_registry (
            id INTEGER PRIMARY KEY,
            model_id TEXT UNIQUE NOT NULL,
            model_name TEXT NOT NULL,
            version TEXT NOT NULL,
            model_type TEXT NOT NULL,
            status TEXT CHECK(status IN ('TRAINING', 'VALIDATION', 'PRODUCTION', 'RETIRED')),
            created_at DATETIME NOT NULL,
            deployed_at DATETIME,
            performance_metrics TEXT,
            artifact_location TEXT
        );
        
        CREATE TABLE model_performance (
            id INTEGER PRIMARY KEY,
            model_id TEXT NOT NULL,
            product_category TEXT NOT NULL,
            timestamp DATETIME NOT NULL,
            accuracy_score REAL,
            mape_score REAL,
            rmse_score REAL,
            prediction_count INTEGER,
            drift_score REAL
        );
        
        CREATE TABLE drift_events (
            id INTEGER PRIMARY KEY,
            model_id TEXT NOT NULL,
            severity TEXT CHECK(severity IN ('LOW', 'MEDIUM', 'HIGH')),
            detected_at DATETIME NOT NULL,
            resolved_at DATETIME,
            accuracy_drop REAL,
            affected_categories TEXT
        );
        
        CREATE TABLE retraining_workflows (
            id INTEGER PRIMARY KEY,
            workflow_id TEXT UNIQUE NOT NULL,
            trigger_reason TEXT NOT NULL,
            started_at DATETIME NOT NULL,
            completed_at DATETIME,
            status TEXT CHECK(status IN ('RUNNING', 'COMPLETED', 'FAILED', 'CANCELLED')),
            models_trained INTEGER DEFAULT 0,
            models_deployed INTEGER DEFAULT 0,
            business_impact_score REAL
        );
        
        CREATE TABLE deployments (
            id INTEGER PRIMARY KEY,
            deployment_id TEXT UNIQUE NOT NULL,
            model_id TEXT NOT NULL,
            deployment_strategy TEXT DEFAULT 'blue_green',
            status TEXT CHECK(status IN ('PENDING', 'DEPLOYING', 'ACTIVE', 'ROLLED_BACK', 'FAILED')),
            started_at DATETIME NOT NULL,
            completed_at DATETIME,
            rollback_at DATETIME,
            performance_metrics TEXT,
            deployment_notes TEXT
        );
        
        CREATE TABLE business_impact (
            id INTEGER PRIMARY KEY,
            deployment_id TEXT NOT NULL,
            metric_type TEXT NOT NULL,
            baseline_value REAL,
            improved_value REAL,
            improvement_percentage REAL,
            revenue_impact REAL,
            calculated_at DATETIME NOT NULL,
            FOREIGN KEY (deployment_id) REFERENCES deployments(deployment_id)
        );
    """)
    
    # Insert sample data
    now = datetime.now()
    
    # Sample models
    models = [
        ('electronics_forecast_v1', 'Electronics Demand Forecast', '1.0', 'XGBoost', 'PRODUCTION'),
        ('clothing_forecast_v1', 'Clothing Demand Forecast', '1.0', 'Prophet', 'PRODUCTION'),
        ('home_garden_forecast_v1', 'Home & Garden Forecast', '1.0', 'LSTM', 'PRODUCTION'),
        ('books_forecast_v2', 'Books Demand Forecast', '2.0', 'Ensemble', 'VALIDATION')
    ]
    
    for i, (model_id, name, version, model_type, status) in enumerate(models):
        cursor.execute("""
            INSERT INTO model_registry 
            (model_id, model_name, version, model_type, status, created_at, deployed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            model_id, name, version, model_type, status,
            now - timedelta(days=30-i*5),
            now - timedelta(days=25-i*5) if status == 'PRODUCTION' else None
        ))
    
    # Sample performance data (last 24 hours)
    categories = ['electronics', 'clothing', 'home_garden', 'books']
    model_ids = [m[0] for m in models]
    
    for hour in range(24):
        timestamp = now - timedelta(hours=hour)
        
        for i, (model_id, category) in enumerate(zip(model_ids, categories)):
            # Simulate accuracy degradation for some models
            base_accuracy = 0.92 - (i * 0.02)
            if model_id == 'clothing_forecast_v1' and hour < 6:
                # Simulate drift in clothing model
                accuracy = base_accuracy - (6-hour) * 0.03
            else:
                # Normal variation
                import random
                accuracy = base_accuracy + random.uniform(-0.02, 0.02)
            
            cursor.execute("""
                INSERT INTO model_performance 
                (model_id, product_category, timestamp, accuracy_score, mape_score, rmse_score, prediction_count)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                model_id, category, timestamp, accuracy,
                (1 - accuracy) * 0.8,  # MAPE
                (1 - accuracy) * 20,   # RMSE
                random.randint(800, 1200)  # Prediction count
            ))
    
    # Sample drift events
    cursor.execute("""
        INSERT INTO drift_events 
        (model_id, severity, detected_at, accuracy_drop, affected_categories)
        VALUES 
        ('clothing_forecast_v1', 'HIGH', ?, 0.18, 'clothing'),
        ('electronics_forecast_v1', 'LOW', ?, 0.04, 'electronics')
    """, (now - timedelta(hours=3), now - timedelta(hours=8)))
    
    # Sample retraining workflows
    cursor.execute("""
        INSERT INTO retraining_workflows 
        (workflow_id, trigger_reason, started_at, status, models_trained)
        VALUES 
        ('workflow_clothing_001', 'High drift detected in clothing forecast', ?, 'RUNNING', 0),
        ('workflow_electronics_002', 'Scheduled retraining', ?, 'COMPLETED', 1)
    """, (now - timedelta(hours=2), now - timedelta(hours=12)))
    
    # Sample deployments
    cursor.execute("""
        INSERT INTO deployments 
        (deployment_id, model_id, status, started_at, completed_at)
        VALUES 
        ('deploy_electronics_001', 'electronics_forecast_v1', 'ACTIVE', ?, ?),
        ('deploy_clothing_001', 'clothing_forecast_v1', 'ACTIVE', ?, ?),
        ('deploy_books_001', 'books_forecast_v2', 'FAILED', ?, NULL)
    """, (
        now - timedelta(days=25), now - timedelta(days=25, hours=2),
        now - timedelta(days=20), now - timedelta(days=20, hours=1),
        now - timedelta(hours=1)
    ))
    
    # Sample business impact
    cursor.execute("""
        INSERT INTO business_impact 
        (deployment_id, metric_type, baseline_value, improved_value, improvement_percentage, revenue_impact, calculated_at)
        VALUES 
        ('deploy_electronics_001', 'accuracy_improvement', 0.85, 0.92, 8.2, 45000.0, ?),
        ('deploy_clothing_001', 'accuracy_improvement', 0.82, 0.88, 7.3, 32000.0, ?)
    """, (now - timedelta(days=25), now - timedelta(days=20)))
    
    conn.commit()
    conn.close()
    
    print(f"✓ Demo database created at: {db_path}")
    print("✓ Sample data inserted:")
    print("  - 4 models (3 in production, 1 in validation)")
    print("  - 24 hours of performance data")
    print("  - 2 drift events (1 high severity, 1 low severity)")
    print("  - 2 retraining workflows")
    print("  - 3 deployments (2 active, 1 failed)")
    print("  - Business impact data")

def main():
    """Run the dashboard demo"""
    print("🚀 Starting Autonomous Demand Forecasting Dashboard Demo")
    print("=" * 60)
    
    # Setup demo database
    demo_db_path = "demo_dashboard.db"
    setup_demo_database(demo_db_path)
    
    print("\n📊 Starting dashboard server...")
    print("Dashboard will be available at: http://localhost:5000")
    print("Press Ctrl+C to stop the server")
    print("-" * 60)
    
    # Create and start dashboard
    dashboard = MonitoringDashboard(db_path=demo_db_path)
    
    try:
        dashboard.start_dashboard(host='localhost', port=5000, debug=True)
    except KeyboardInterrupt:
        print("\n\n🛑 Stopping dashboard...")
        dashboard.stop_dashboard()
        print("✓ Dashboard stopped")
    finally:
        # Clean up demo database
        if os.path.exists(demo_db_path):
            os.remove(demo_db_path)
            print("✓ Demo database cleaned up")

if __name__ == "__main__":
    main()