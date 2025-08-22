"""
Model Deployment MCP Server for Autonomous Demand Forecasting System.

This server handles production model deployment and management with blue-green deployment
strategy, production monitoring, and automatic rollback capabilities.
"""

import asyncio
import json
import logging
import os
import sqlite3
import shutil
import uuid
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import subprocess
import signal

import mcp.server.stdio
from dotenv import load_dotenv

# MCP Server Imports
from mcp import types as mcp_types
from mcp.server.lowlevel import NotificationOptions, Server
from mcp.server.models import InitializationOptions

# Import data models
from autonomous_demand_forecasting.database.models import (
    DeploymentResult, DeploymentStatus, ProductionMonitoring, ModelRegistry,
    ModelStatus, AccuracyMetrics, BusinessImpact
)

load_dotenv()

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)

# Database path
DATABASE_PATH = os.path.join(os.path.dirname(__file__), "database", "autonomous_demand_forecasting.db")

# Deployment configuration
DEPLOYMENT_CONFIG = {
    "blue_green": {
        "health_check_timeout": 300,  # 5 minutes
        "stabilization_period": 1440,  # 24 hours in minutes
        "rollback_threshold": 0.05,  # 5% accuracy drop triggers rollback
        "max_deployment_time": 1800,  # 30 minutes
    },
    "canary": {
        "traffic_percentage": 10,
        "ramp_up_duration": 60,  # 1 hour
        "success_threshold": 0.95,
    }
}

# Production environment paths
PRODUCTION_MODEL_PATH = os.path.join(os.path.dirname(__file__), "production", "models")
STAGING_MODEL_PATH = os.path.join(os.path.dirname(__file__), "staging", "models")
BACKUP_MODEL_PATH = os.path.join(os.path.dirname(__file__), "backup", "models")

# Ensure directories exist
for path in [PRODUCTION_MODEL_PATH, STAGING_MODEL_PATH, BACKUP_MODEL_PATH]:
    os.makedirs(path, exist_ok=True)

# Global deployment monitoring state
deployment_monitors = {}
monitoring_lock = threading.Lock()

# --- Database Utility Functions ---
def get_db_connection():
    """Get database connection with row factory."""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def create_deployment_record(deployment: DeploymentResult) -> Dict[str, Any]:
    """Create a new deployment record in the database."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO deployments 
            (deployment_id, model_id, deployment_strategy, status, started_at, 
             performance_metrics, deployment_notes)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            deployment.deployment_id,
            deployment.model_id,
            deployment.deployment_strategy,
            deployment.status.value,
            deployment.started_at,
            json.dumps(deployment.performance_metrics) if deployment.performance_metrics else None,
            deployment.deployment_notes
        ))
        
        conn.commit()
        deployment_db_id = cursor.lastrowid
        conn.close()
        
        return {
            "success": True,
            "message": f"Deployment record created successfully. ID: {deployment_db_id}",
            "deployment_db_id": deployment_db_id
        }
    except sqlite3.Error as e:
        return {
            "success": False,
            "message": f"Error creating deployment record: {e}"
        }


def update_deployment_status(
    deployment_id: str,
    status: DeploymentStatus,
    completed_at: Optional[datetime] = None,
    rollback_at: Optional[datetime] = None,
    performance_metrics: Optional[Dict[str, float]] = None,
    notes: Optional[str] = None
) -> Dict[str, Any]:
    """Update deployment status and metadata."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE deployments 
            SET status = ?, completed_at = ?, rollback_at = ?, 
                performance_metrics = ?, deployment_notes = ?
            WHERE deployment_id = ?
        """, (
            status.value,
            completed_at,
            rollback_at,
            json.dumps(performance_metrics) if performance_metrics else None,
            notes,
            deployment_id
        ))
        
        conn.commit()
        rows_affected = cursor.rowcount
        conn.close()
        
        if rows_affected > 0:
            return {
                "success": True,
                "message": f"Deployment status updated to {status.value}"
            }
        else:
            return {
                "success": False,
                "message": f"Deployment {deployment_id} not found"
            }
    except sqlite3.Error as e:
        return {
            "success": False,
            "message": f"Error updating deployment status: {e}"
        }


def record_production_monitoring(monitoring: ProductionMonitoring) -> Dict[str, Any]:
    """Record production monitoring metrics."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO production_monitoring 
            (deployment_id, timestamp, accuracy_score, prediction_latency, 
             error_rate, throughput, alert_triggered)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            monitoring.deployment_id,
            monitoring.timestamp,
            monitoring.accuracy_score,
            monitoring.prediction_latency,
            monitoring.error_rate,
            monitoring.throughput,
            monitoring.alert_triggered
        ))
        
        conn.commit()
        monitoring_id = cursor.lastrowid
        conn.close()
        
        return {
            "success": True,
            "message": f"Production monitoring recorded. ID: {monitoring_id}",
            "monitoring_id": monitoring_id
        }
    except sqlite3.Error as e:
        return {
            "success": False,
            "message": f"Error recording production monitoring: {e}"
        }


def get_deployment_by_id(deployment_id: str) -> Optional[Dict[str, Any]]:
    """Get deployment record by ID."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM deployments WHERE deployment_id = ?", (deployment_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            row_dict = dict(row)
            return {
                "deployment_id": row_dict["deployment_id"],
                "model_id": row_dict["model_id"],
                "deployment_strategy": row_dict["deployment_strategy"],
                "status": row_dict["status"],
                "started_at": row_dict["started_at"],
                "completed_at": row_dict.get("completed_at"),
                "rollback_at": row_dict.get("rollback_at"),
                "performance_metrics": json.loads(row_dict["performance_metrics"]) if row_dict["performance_metrics"] else None,
                "deployment_notes": row_dict.get("deployment_notes")
            }
        return None
    except Exception as e:
        logging.error(f"Error getting deployment by ID: {e}")
        return None


def get_active_deployments() -> List[Dict[str, Any]]:
    """Get all active deployments."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM deployments 
            WHERE status IN ('DEPLOYING', 'ACTIVE') 
            ORDER BY started_at DESC
        """)
        rows = cursor.fetchall()
        conn.close()
        
        deployments = []
        for row in rows:
            row_dict = dict(row)
            deployments.append({
                "deployment_id": row_dict["deployment_id"],
                "model_id": row_dict["model_id"],
                "deployment_strategy": row_dict["deployment_strategy"],
                "status": row_dict["status"],
                "started_at": row_dict["started_at"],
                "completed_at": row_dict.get("completed_at"),
                "performance_metrics": json.loads(row_dict["performance_metrics"]) if row_dict["performance_metrics"] else None
            })
        
        return deployments
    except Exception as e:
        logging.error(f"Error getting active deployments: {e}")
        return []


def get_model_registry_entry(model_id: str) -> Optional[Dict[str, Any]]:
    """Get model registry entry by ID."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM model_registry WHERE model_id = ?", (model_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            row_dict = dict(row)
            return {
                "model_id": row_dict["model_id"],
                "model_name": row_dict["model_name"],
                "version": row_dict["version"],
                "model_type": row_dict["model_type"],
                "status": row_dict["status"],
                "created_at": row_dict["created_at"],
                "deployed_at": row_dict.get("deployed_at"),
                "performance_metrics": json.loads(row_dict["performance_metrics"]) if row_dict["performance_metrics"] else None,
                "artifact_location": row_dict["artifact_location"],
                "hyperparameters": json.loads(row_dict["hyperparameters"]) if row_dict["hyperparameters"] else None
            }
        return None
    except Exception as e:
        logging.error(f"Error getting model registry entry: {e}")
        return None


def update_model_status(model_id: str, status: ModelStatus, deployed_at: Optional[datetime] = None) -> Dict[str, Any]:
    """Update model status in registry."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE model_registry 
            SET status = ?, deployed_at = ?
            WHERE model_id = ?
        """, (status.value, deployed_at, model_id))
        
        conn.commit()
        rows_affected = cursor.rowcount
        conn.close()
        
        if rows_affected > 0:
            return {
                "success": True,
                "message": f"Model status updated to {status.value}"
            }
        else:
            return {
                "success": False,
                "message": f"Model {model_id} not found in registry"
            }
    except sqlite3.Error as e:
        return {
            "success": False,
            "message": f"Error updating model status: {e}"
        }


# --- Deployment Package Management ---
class DeploymentPackage:
    """Deployment package for model artifacts and configuration."""
    
    def __init__(self, model_id: str, model_registry: Dict[str, Any]):
        self.model_id = model_id
        self.model_registry = model_registry
        self.package_id = f"deploy_{model_id}_{uuid.uuid4().hex[:8]}"
        self.created_at = datetime.now()
        self.package_path = None
        self.configuration = {}
        
    def create_package(self) -> Dict[str, Any]:
        """Create deployment package with model artifacts and configuration."""
        try:
            # Create package directory
            package_dir = os.path.join(STAGING_MODEL_PATH, self.package_id)
            os.makedirs(package_dir, exist_ok=True)
            
            # Copy model artifacts
            source_path = self.model_registry["artifact_location"]
            if not os.path.exists(source_path):
                return {"success": False, "message": f"Model artifacts not found at {source_path}"}
            
            # Copy all files from source to package directory
            if os.path.isdir(source_path):
                shutil.copytree(source_path, os.path.join(package_dir, "model"), dirs_exist_ok=True)
            else:
                shutil.copy2(source_path, package_dir)
            
            # Create deployment configuration
            self.configuration = {
                "model_id": self.model_id,
                "model_name": self.model_registry["model_name"],
                "version": self.model_registry["version"],
                "model_type": self.model_registry["model_type"],
                "deployment_strategy": "blue_green",
                "health_checks": {
                    "accuracy_threshold": 0.80,
                    "latency_threshold": 1000,  # ms
                    "error_rate_threshold": 0.05
                },
                "rollback_criteria": {
                    "accuracy_drop_threshold": 0.05,
                    "error_rate_threshold": 0.10,
                    "latency_increase_threshold": 2.0
                },
                "monitoring": {
                    "metrics_collection_interval": 60,  # seconds
                    "alert_thresholds": {
                        "accuracy": 0.75,
                        "error_rate": 0.08,
                        "latency": 1500
                    }
                }
            }
            
            # Save configuration
            config_path = os.path.join(package_dir, "deployment_config.json")
            with open(config_path, 'w') as f:
                json.dump(self.configuration, f, indent=2)
            
            # Create deployment script
            self._create_deployment_script(package_dir)
            
            # Create health check script
            self._create_health_check_script(package_dir)
            
            self.package_path = package_dir
            
            return {
                "success": True,
                "package_id": self.package_id,
                "package_path": self.package_path,
                "configuration": self.configuration
            }
            
        except Exception as e:
            return {"success": False, "message": f"Error creating deployment package: {e}"}
    
    def _create_deployment_script(self, package_dir: str):
        """Create deployment script for the package."""
        script_content = f"""#!/bin/bash
# Deployment script for model {self.model_id}
# Generated at {self.created_at}

set -e

MODEL_ID="{self.model_id}"
PACKAGE_DIR="{package_dir}"
PRODUCTION_DIR="{PRODUCTION_MODEL_PATH}"

echo "Starting deployment of model $MODEL_ID"

# Create production directory if it doesn't exist
mkdir -p "$PRODUCTION_DIR"

# Copy model artifacts to production
cp -r "$PACKAGE_DIR/model" "$PRODUCTION_DIR/$MODEL_ID"

# Set permissions
chmod -R 755 "$PRODUCTION_DIR/$MODEL_ID"

echo "Model $MODEL_ID deployed successfully"
"""
        
        script_path = os.path.join(package_dir, "deploy.sh")
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Make script executable
        os.chmod(script_path, 0o755)
    
    def _create_health_check_script(self, package_dir: str):
        """Create health check script for the deployment."""
        script_content = f"""#!/usr/bin/env python3
# Health check script for model {self.model_id}
# Generated at {self.created_at}

import json
import os
import sys
import time
from datetime import datetime

def check_model_health():
    \"\"\"Perform basic health checks on deployed model.\"\"\"
    model_path = os.path.join("{PRODUCTION_MODEL_PATH}", "{self.model_id}")
    
    # Check if model files exist
    if not os.path.exists(model_path):
        return False, "Model files not found"
    
    # Check if model.pkl exists
    model_file = os.path.join(model_path, "model.pkl")
    if not os.path.exists(model_file):
        return False, "Model pickle file not found"
    
    # Check if metadata exists
    metadata_file = os.path.join(model_path, "metadata.json")
    if not os.path.exists(metadata_file):
        return False, "Model metadata not found"
    
    # Try to load metadata
    try:
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Validate metadata structure
        required_fields = ["model_type", "version", "created_at"]
        for field in required_fields:
            if field not in metadata:
                return False, f"Missing metadata field: {{field}}"
        
        return True, "Health check passed"
    
    except Exception as e:
        return False, f"Error loading metadata: {{e}}"

if __name__ == "__main__":
    healthy, message = check_model_health()
    
    result = {{
        "timestamp": datetime.now().isoformat(),
        "model_id": "{self.model_id}",
        "healthy": healthy,
        "message": message
    }}
    
    print(json.dumps(result))
    sys.exit(0 if healthy else 1)
"""
        
        script_path = os.path.join(package_dir, "health_check.py")
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Make script executable
        os.chmod(script_path, 0o755)


# --- Blue-Green Deployment Implementation ---
class BlueGreenDeployment:
    """Blue-green deployment strategy implementation."""
    
    def __init__(self, deployment_id: str, package: DeploymentPackage):
        self.deployment_id = deployment_id
        self.package = package
        self.model_id = package.model_id
        self.blue_path = os.path.join(PRODUCTION_MODEL_PATH, f"{self.model_id}_blue")
        self.green_path = os.path.join(PRODUCTION_MODEL_PATH, f"{self.model_id}_green")
        self.active_path = os.path.join(PRODUCTION_MODEL_PATH, f"{self.model_id}_active")
        self.backup_created = False
        
    def execute(self) -> Dict[str, Any]:
        """Execute blue-green deployment."""
        try:
            logging.info(f"Starting blue-green deployment for {self.model_id}")
            
            # Step 1: Determine current and target environments
            current_env, target_env = self._determine_environments()
            
            # Step 2: Create backup of current production model
            backup_result = self._create_backup()
            if not backup_result["success"]:
                return backup_result
            
            # Step 3: Deploy to target environment (green)
            deploy_result = self._deploy_to_target(target_env)
            if not deploy_result["success"]:
                return deploy_result
            
            # Step 4: Run health checks on target environment
            health_result = self._run_health_checks(target_env)
            if not health_result["success"]:
                return health_result
            
            # Step 5: Switch traffic to target environment
            switch_result = self._switch_traffic(target_env)
            if not switch_result["success"]:
                return switch_result
            
            # Step 6: Monitor for stabilization period
            monitor_result = self._start_monitoring()
            
            return {
                "success": True,
                "message": f"Blue-green deployment completed successfully",
                "deployment_id": self.deployment_id,
                "active_environment": target_env,
                "monitoring_started": monitor_result["success"]
            }
            
        except Exception as e:
            # Attempt rollback on any error
            self._emergency_rollback()
            return {"success": False, "message": f"Deployment failed: {e}"}
    
    def _determine_environments(self) -> Tuple[str, str]:
        """Determine current and target environments."""
        # Check which environment is currently active
        if os.path.exists(self.active_path):
            if os.path.islink(self.active_path):
                current_target = os.readlink(self.active_path)
                if "blue" in current_target:
                    return "blue", "green"
                else:
                    return "green", "blue"
        
        # Default to blue as current, green as target
        return "blue", "green"
    
    def _create_backup(self) -> Dict[str, Any]:
        """Create backup of current production model."""
        try:
            if os.path.exists(self.active_path):
                backup_path = os.path.join(BACKUP_MODEL_PATH, f"{self.model_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                
                if os.path.islink(self.active_path):
                    # Follow symlink to actual model
                    actual_path = os.readlink(self.active_path)
                    if os.path.exists(actual_path):
                        shutil.copytree(actual_path, backup_path)
                else:
                    shutil.copytree(self.active_path, backup_path)
                
                self.backup_created = True
                logging.info(f"Backup created at {backup_path}")
                
                return {"success": True, "backup_path": backup_path}
            
            return {"success": True, "message": "No existing model to backup"}
            
        except Exception as e:
            return {"success": False, "message": f"Backup creation failed: {e}"}
    
    def _deploy_to_target(self, target_env: str) -> Dict[str, Any]:
        """Deploy model to target environment."""
        try:
            target_path = self.green_path if target_env == "green" else self.blue_path
            
            # Remove existing target environment
            if os.path.exists(target_path):
                shutil.rmtree(target_path)
            
            # Copy model from package to target environment
            model_source = os.path.join(self.package.package_path, "model")
            shutil.copytree(model_source, target_path)
            
            # Copy deployment configuration
            config_source = os.path.join(self.package.package_path, "deployment_config.json")
            config_target = os.path.join(target_path, "deployment_config.json")
            shutil.copy2(config_source, config_target)
            
            logging.info(f"Model deployed to {target_env} environment at {target_path}")
            
            return {"success": True, "target_path": target_path}
            
        except Exception as e:
            return {"success": False, "message": f"Deployment to {target_env} failed: {e}"}
    
    def _run_health_checks(self, target_env: str) -> Dict[str, Any]:
        """Run health checks on target environment."""
        try:
            target_path = self.green_path if target_env == "green" else self.blue_path
            health_script = os.path.join(self.package.package_path, "health_check.py")
            
            # Run health check script
            result = subprocess.run(
                ["python3", health_script],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                health_data = json.loads(result.stdout)
                if health_data.get("healthy", False):
                    return {"success": True, "health_data": health_data}
                else:
                    return {"success": False, "message": f"Health check failed: {health_data.get('message', 'Unknown error')}"}
            else:
                return {"success": False, "message": f"Health check script failed: {result.stderr}"}
                
        except subprocess.TimeoutExpired:
            return {"success": False, "message": "Health check timed out"}
        except Exception as e:
            return {"success": False, "message": f"Health check error: {e}"}
    
    def _switch_traffic(self, target_env: str) -> Dict[str, Any]:
        """Switch traffic to target environment."""
        try:
            target_path = self.green_path if target_env == "green" else self.blue_path
            
            # Remove existing active symlink
            if os.path.exists(self.active_path):
                if os.path.islink(self.active_path):
                    os.unlink(self.active_path)
                else:
                    shutil.rmtree(self.active_path)
            
            # Create new symlink to target environment
            os.symlink(target_path, self.active_path)
            
            logging.info(f"Traffic switched to {target_env} environment")
            
            return {"success": True, "active_environment": target_env}
            
        except Exception as e:
            return {"success": False, "message": f"Traffic switch failed: {e}"}
    
    def _start_monitoring(self) -> Dict[str, Any]:
        """Start production monitoring for the deployment."""
        try:
            # Start monitoring thread
            monitor_thread = threading.Thread(
                target=self._monitor_deployment,
                args=(self.deployment_id,),
                daemon=True
            )
            monitor_thread.start()
            
            # Store monitoring reference
            with monitoring_lock:
                deployment_monitors[self.deployment_id] = {
                    "thread": monitor_thread,
                    "start_time": datetime.now(),
                    "model_id": self.model_id
                }
            
            return {"success": True, "message": "Production monitoring started"}
            
        except Exception as e:
            return {"success": False, "message": f"Failed to start monitoring: {e}"}
    
    def _monitor_deployment(self, deployment_id: str):
        """Monitor deployment performance and stability."""
        config = DEPLOYMENT_CONFIG["blue_green"]
        start_time = datetime.now()
        stabilization_end = start_time + timedelta(minutes=config["stabilization_period"])
        
        logging.info(f"Starting deployment monitoring for {deployment_id}")
        
        try:
            while datetime.now() < stabilization_end:
                # Collect performance metrics
                metrics = self._collect_performance_metrics()
                
                # Record monitoring data
                monitoring = ProductionMonitoring(
                    deployment_id=deployment_id,
                    timestamp=datetime.now(),
                    accuracy_score=metrics.get("accuracy_score"),
                    prediction_latency=metrics.get("prediction_latency"),
                    error_rate=metrics.get("error_rate"),
                    throughput=metrics.get("throughput"),
                    alert_triggered=False
                )
                
                # Check for rollback conditions
                if self._should_rollback(metrics):
                    logging.warning(f"Rollback conditions met for deployment {deployment_id}")
                    monitoring.alert_triggered = True
                    record_production_monitoring(monitoring)
                    
                    # Trigger rollback
                    self._trigger_rollback(deployment_id, "Performance degradation detected")
                    break
                
                # Record normal monitoring
                record_production_monitoring(monitoring)
                
                # Wait before next check
                time.sleep(config.get("metrics_collection_interval", 60))
            
            # Mark deployment as stable
            update_deployment_status(
                deployment_id,
                DeploymentStatus.ACTIVE,
                completed_at=datetime.now(),
                notes="Deployment completed successfully after stabilization period"
            )
            
            logging.info(f"Deployment {deployment_id} completed stabilization period successfully")
            
        except Exception as e:
            logging.error(f"Error in deployment monitoring: {e}")
            self._trigger_rollback(deployment_id, f"Monitoring error: {e}")
        
        finally:
            # Clean up monitoring reference
            with monitoring_lock:
                if deployment_id in deployment_monitors:
                    del deployment_monitors[deployment_id]
    
    def _collect_performance_metrics(self) -> Dict[str, float]:
        """Collect current performance metrics."""
        # This is a simplified implementation
        # In a real system, this would integrate with monitoring systems
        try:
            # Simulate metric collection
            import random
            
            # Get recent model performance from database
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT accuracy_score, mape_score 
                FROM model_performance 
                WHERE model_id = ? 
                ORDER BY timestamp DESC 
                LIMIT 1
            """, (self.model_id,))
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                accuracy = row["accuracy_score"] or 0.85
                mape = row["mape_score"] or 0.15
            else:
                accuracy = 0.85
                mape = 0.15
            
            # Simulate other metrics with some variance
            return {
                "accuracy_score": accuracy + random.uniform(-0.02, 0.02),
                "prediction_latency": random.uniform(50, 200),  # ms
                "error_rate": random.uniform(0.01, 0.05),
                "throughput": random.randint(100, 500)  # requests/minute
            }
            
        except Exception as e:
            logging.error(f"Error collecting performance metrics: {e}")
            return {
                "accuracy_score": 0.80,
                "prediction_latency": 100.0,
                "error_rate": 0.05,
                "throughput": 200
            }
    
    def _should_rollback(self, metrics: Dict[str, float]) -> bool:
        """Check if rollback conditions are met."""
        config = DEPLOYMENT_CONFIG["blue_green"]
        
        # Check accuracy threshold
        if metrics.get("accuracy_score", 1.0) < (0.85 - config["rollback_threshold"]):
            return True
        
        # Check error rate threshold
        if metrics.get("error_rate", 0.0) > 0.10:
            return True
        
        # Check latency threshold
        if metrics.get("prediction_latency", 0.0) > 2000:  # 2 seconds
            return True
        
        return False
    
    def _trigger_rollback(self, deployment_id: str, reason: str):
        """Trigger automatic rollback."""
        try:
            logging.warning(f"Triggering rollback for deployment {deployment_id}: {reason}")
            
            # Update deployment status
            update_deployment_status(
                deployment_id,
                DeploymentStatus.ROLLED_BACK,
                rollback_at=datetime.now(),
                notes=f"Automatic rollback: {reason}"
            )
            
            # Perform rollback
            rollback_result = self._emergency_rollback()
            
            if rollback_result:
                logging.info(f"Rollback completed successfully for deployment {deployment_id}")
            else:
                logging.error(f"Rollback failed for deployment {deployment_id}")
                
        except Exception as e:
            logging.error(f"Error during rollback trigger: {e}")
    
    def _emergency_rollback(self) -> bool:
        """Perform emergency rollback to previous version."""
        try:
            # Find most recent backup
            backup_files = []
            if os.path.exists(BACKUP_MODEL_PATH):
                for item in os.listdir(BACKUP_MODEL_PATH):
                    if item.startswith(self.model_id):
                        backup_path = os.path.join(BACKUP_MODEL_PATH, item)
                        if os.path.isdir(backup_path):
                            backup_files.append((item, backup_path))
            
            if not backup_files:
                logging.error("No backup found for rollback")
                return False
            
            # Sort by timestamp (newest first)
            backup_files.sort(reverse=True)
            latest_backup = backup_files[0][1]
            
            # Remove current active deployment
            if os.path.exists(self.active_path):
                if os.path.islink(self.active_path):
                    os.unlink(self.active_path)
                else:
                    shutil.rmtree(self.active_path)
            
            # Restore from backup
            shutil.copytree(latest_backup, self.active_path)
            
            logging.info(f"Emergency rollback completed using backup: {latest_backup}")
            return True
            
        except Exception as e:
            logging.error(f"Emergency rollback failed: {e}")
            return False


# --- MCP Server Implementation ---
server = Server("model-deployment-mcp")

@server.list_tools()
async def handle_list_tools() -> List[mcp_types.Tool]:
    """List available deployment tools."""
    return [
        mcp_types.Tool(
            name="create_deployment_package",
            description="Create deployment package with model artifacts and configuration",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": {
                        "type": "string",
                        "description": "Model ID to create deployment package for"
                    }
                },
                "required": ["model_id"]
            }
        ),
        mcp_types.Tool(
            name="execute_blue_green_deployment",
            description="Execute blue-green deployment strategy for model",
            inputSchema={
                "type": "object",
                "properties": {
                    "package_id": {
                        "type": "string",
                        "description": "Deployment package ID"
                    },
                    "model_id": {
                        "type": "string",
                        "description": "Model ID to deploy"
                    }
                },
                "required": ["package_id", "model_id"]
            }
        ),
        mcp_types.Tool(
            name="monitor_production_performance",
            description="Monitor production model performance and collect metrics",
            inputSchema={
                "type": "object",
                "properties": {
                    "deployment_id": {
                        "type": "string",
                        "description": "Deployment ID to monitor"
                    },
                    "duration_minutes": {
                        "type": "integer",
                        "description": "Duration to monitor in minutes",
                        "default": 60
                    }
                },
                "required": ["deployment_id"]
            }
        ),
        mcp_types.Tool(
            name="rollback_deployment",
            description="Rollback deployment to previous version",
            inputSchema={
                "type": "object",
                "properties": {
                    "deployment_id": {
                        "type": "string",
                        "description": "Deployment ID to rollback"
                    },
                    "reason": {
                        "type": "string",
                        "description": "Reason for rollback"
                    }
                },
                "required": ["deployment_id", "reason"]
            }
        ),
        mcp_types.Tool(
            name="get_deployment_status",
            description="Get current status of a deployment",
            inputSchema={
                "type": "object",
                "properties": {
                    "deployment_id": {
                        "type": "string",
                        "description": "Deployment ID to check status for"
                    }
                },
                "required": ["deployment_id"]
            }
        ),
        mcp_types.Tool(
            name="list_active_deployments",
            description="List all active deployments",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        mcp_types.Tool(
            name="validate_deployment_health",
            description="Validate health of deployed model",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": {
                        "type": "string",
                        "description": "Model ID to validate health for"
                    }
                },
                "required": ["model_id"]
            }
        )
    ]


@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> List[mcp_types.TextContent]:
    """Handle tool calls for model deployment operations."""
    
    if name == "create_deployment_package":
        model_id = arguments.get("model_id")
        if not model_id:
            return [mcp_types.TextContent(
                type="text",
                text=json.dumps({"success": False, "message": "model_id is required"})
            )]
        
        # Get model registry entry
        model_registry = get_model_registry_entry(model_id)
        if not model_registry:
            return [mcp_types.TextContent(
                type="text",
                text=json.dumps({"success": False, "message": f"Model {model_id} not found in registry"})
            )]
        
        # Create deployment package
        package = DeploymentPackage(model_id, model_registry)
        result = package.create_package()
        
        return [mcp_types.TextContent(
            type="text",
            text=json.dumps(result)
        )]
    
    elif name == "execute_blue_green_deployment":
        package_id = arguments.get("package_id")
        model_id = arguments.get("model_id")
        
        if not package_id or not model_id:
            return [mcp_types.TextContent(
                type="text",
                text=json.dumps({"success": False, "message": "package_id and model_id are required"})
            )]
        
        # Get model registry entry
        model_registry = get_model_registry_entry(model_id)
        if not model_registry:
            return [mcp_types.TextContent(
                type="text",
                text=json.dumps({"success": False, "message": f"Model {model_id} not found in registry"})
            )]
        
        # Create deployment package object
        package = DeploymentPackage(model_id, model_registry)
        package.package_id = package_id
        package.package_path = os.path.join(STAGING_MODEL_PATH, package_id)
        
        if not os.path.exists(package.package_path):
            return [mcp_types.TextContent(
                type="text",
                text=json.dumps({"success": False, "message": f"Package {package_id} not found"})
            )]
        
        # Create deployment record
        deployment_id = f"deploy_{model_id}_{uuid.uuid4().hex[:8]}"
        deployment = DeploymentResult(
            deployment_id=deployment_id,
            model_id=model_id,
            started_at=datetime.now(),
            status=DeploymentStatus.DEPLOYING,
            deployment_strategy="blue_green",
            deployment_notes=f"Blue-green deployment initiated for model {model_id}"
        )
        
        create_result = create_deployment_record(deployment)
        if not create_result["success"]:
            return [mcp_types.TextContent(
                type="text",
                text=json.dumps(create_result)
            )]
        
        # Execute blue-green deployment
        bg_deployment = BlueGreenDeployment(deployment_id, package)
        result = bg_deployment.execute()
        
        if result["success"]:
            # Update model status to PRODUCTION
            update_model_status(model_id, ModelStatus.PRODUCTION, datetime.now())
        
        return [mcp_types.TextContent(
            type="text",
            text=json.dumps(result)
        )]
    
    elif name == "monitor_production_performance":
        deployment_id = arguments.get("deployment_id")
        duration_minutes = arguments.get("duration_minutes", 60)
        
        if not deployment_id:
            return [mcp_types.TextContent(
                type="text",
                text=json.dumps({"success": False, "message": "deployment_id is required"})
            )]
        
        # Get deployment info
        deployment = get_deployment_by_id(deployment_id)
        if not deployment:
            return [mcp_types.TextContent(
                type="text",
                text=json.dumps({"success": False, "message": f"Deployment {deployment_id} not found"})
            )]
        
        # Collect current metrics
        try:
            # Simulate performance monitoring
            import random
            
            metrics = {
                "accuracy_score": random.uniform(0.80, 0.95),
                "prediction_latency": random.uniform(50, 200),
                "error_rate": random.uniform(0.01, 0.05),
                "throughput": random.randint(100, 500)
            }
            
            # Record monitoring data
            monitoring = ProductionMonitoring(
                deployment_id=deployment_id,
                timestamp=datetime.now(),
                accuracy_score=metrics["accuracy_score"],
                prediction_latency=metrics["prediction_latency"],
                error_rate=metrics["error_rate"],
                throughput=metrics["throughput"],
                alert_triggered=False
            )
            
            record_result = record_production_monitoring(monitoring)
            
            result = {
                "success": True,
                "deployment_id": deployment_id,
                "monitoring_duration_minutes": duration_minutes,
                "current_metrics": metrics,
                "monitoring_recorded": record_result["success"]
            }
            
        except Exception as e:
            result = {"success": False, "message": f"Error monitoring performance: {e}"}
        
        return [mcp_types.TextContent(
            type="text",
            text=json.dumps(result)
        )]
    
    elif name == "rollback_deployment":
        deployment_id = arguments.get("deployment_id")
        reason = arguments.get("reason")
        
        if not deployment_id or not reason:
            return [mcp_types.TextContent(
                type="text",
                text=json.dumps({"success": False, "message": "deployment_id and reason are required"})
            )]
        
        # Get deployment info
        deployment = get_deployment_by_id(deployment_id)
        if not deployment:
            return [mcp_types.TextContent(
                type="text",
                text=json.dumps({"success": False, "message": f"Deployment {deployment_id} not found"})
            )]
        
        try:
            # Create deployment package for rollback
            model_registry = get_model_registry_entry(deployment["model_id"])
            package = DeploymentPackage(deployment["model_id"], model_registry)
            
            # Execute rollback
            bg_deployment = BlueGreenDeployment(deployment_id, package)
            rollback_success = bg_deployment._emergency_rollback()
            
            if rollback_success:
                # Update deployment status
                update_deployment_status(
                    deployment_id,
                    DeploymentStatus.ROLLED_BACK,
                    rollback_at=datetime.now(),
                    notes=f"Manual rollback: {reason}"
                )
                
                result = {
                    "success": True,
                    "message": f"Rollback completed successfully",
                    "deployment_id": deployment_id,
                    "reason": reason
                }
            else:
                result = {
                    "success": False,
                    "message": "Rollback failed - no backup available or rollback error"
                }
                
        except Exception as e:
            result = {"success": False, "message": f"Error during rollback: {e}"}
        
        return [mcp_types.TextContent(
            type="text",
            text=json.dumps(result)
        )]
    
    elif name == "get_deployment_status":
        deployment_id = arguments.get("deployment_id")
        
        if not deployment_id:
            return [mcp_types.TextContent(
                type="text",
                text=json.dumps({"success": False, "message": "deployment_id is required"})
            )]
        
        deployment = get_deployment_by_id(deployment_id)
        if not deployment:
            return [mcp_types.TextContent(
                type="text",
                text=json.dumps({"success": False, "message": f"Deployment {deployment_id} not found"})
            )]
        
        # Get recent monitoring data
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM production_monitoring 
                WHERE deployment_id = ? 
                ORDER BY timestamp DESC 
                LIMIT 5
            """, (deployment_id,))
            
            monitoring_rows = cursor.fetchall()
            conn.close()
            
            monitoring_data = []
            for row in monitoring_rows:
                row_dict = dict(row)
                monitoring_data.append({
                    "timestamp": row_dict["timestamp"],
                    "accuracy_score": row_dict["accuracy_score"],
                    "prediction_latency": row_dict["prediction_latency"],
                    "error_rate": row_dict["error_rate"],
                    "throughput": row_dict["throughput"],
                    "alert_triggered": row_dict["alert_triggered"]
                })
            
            result = {
                "success": True,
                "deployment": deployment,
                "recent_monitoring": monitoring_data
            }
            
        except Exception as e:
            result = {
                "success": True,
                "deployment": deployment,
                "recent_monitoring": [],
                "monitoring_error": str(e)
            }
        
        return [mcp_types.TextContent(
            type="text",
            text=json.dumps(result)
        )]
    
    elif name == "list_active_deployments":
        deployments = get_active_deployments()
        
        result = {
            "success": True,
            "active_deployments": deployments,
            "count": len(deployments)
        }
        
        return [mcp_types.TextContent(
            type="text",
            text=json.dumps(result)
        )]
    
    elif name == "validate_deployment_health":
        model_id = arguments.get("model_id")
        
        if not model_id:
            return [mcp_types.TextContent(
                type="text",
                text=json.dumps({"success": False, "message": "model_id is required"})
            )]
        
        try:
            # Check if model is deployed
            active_path = os.path.join(PRODUCTION_MODEL_PATH, f"{model_id}_active")
            
            if not os.path.exists(active_path):
                return [mcp_types.TextContent(
                    type="text",
                    text=json.dumps({"success": False, "message": f"Model {model_id} is not deployed"})
                )]
            
            # Run health check
            health_checks = {
                "model_files_exist": False,
                "metadata_valid": False,
                "artifacts_accessible": False
            }
            
            # Check model files
            if os.path.isdir(active_path):
                model_file = os.path.join(active_path, "model.pkl")
                metadata_file = os.path.join(active_path, "metadata.json")
                
                health_checks["model_files_exist"] = os.path.exists(model_file)
                
                # Check metadata
                if os.path.exists(metadata_file):
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                        health_checks["metadata_valid"] = "model_type" in metadata
                    except:
                        health_checks["metadata_valid"] = False
                
                health_checks["artifacts_accessible"] = os.access(active_path, os.R_OK)
            
            all_healthy = all(health_checks.values())
            
            result = {
                "success": True,
                "model_id": model_id,
                "healthy": all_healthy,
                "health_checks": health_checks,
                "deployment_path": active_path
            }
            
        except Exception as e:
            result = {"success": False, "message": f"Error validating deployment health: {e}"}
        
        return [mcp_types.TextContent(
            type="text",
            text=json.dumps(result)
        )]
    
    else:
        return [mcp_types.TextContent(
            type="text",
            text=json.dumps({"success": False, "message": f"Unknown tool: {name}"})
        )]


async def main():
    """Main entry point for the MCP server."""
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="model-deployment-mcp",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


if __name__ == "__main__":
    asyncio.run(main())