"""
Forecasting Model MCP Server for Autonomous Demand Forecasting System.

This server handles machine learning model training and optimization for demand forecasting,
implementing multiple forecasting algorithms with automated hyperparameter optimization.
"""

import asyncio
import json
import logging
import os
import sqlite3
import pickle
import hashlib
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import statistics
import numpy as np
import pandas as pd

import mcp.server.stdio
from dotenv import load_dotenv

# MCP Server Imports
from mcp import types as mcp_types
from mcp.server.lowlevel import NotificationOptions, Server
from mcp.server.models import InitializationOptions

# Machine Learning Imports
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
    from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, RandomizedSearchCV
    from sklearn.preprocessing import StandardScaler
    import xgboost as xgb
    ML_AVAILABLE = True
    
    # Try to import Bayesian optimization
    try:
        from skopt import BayesSearchCV
        from skopt.space import Real, Integer, Categorical
        BAYESIAN_OPT_AVAILABLE = True
    except ImportError:
        BAYESIAN_OPT_AVAILABLE = False
        logging.info("Bayesian optimization not available. Install scikit-optimize for advanced hyperparameter tuning.")
        
except ImportError:
    ML_AVAILABLE = False
    BAYESIAN_OPT_AVAILABLE = False
    logging.warning("Machine learning libraries not available. Install scikit-learn, xgboost for full functionality.")

# Import data models
from autonomous_demand_forecasting.database.models import (
    ModelExperiment, ModelRegistry, ModelStatus, TrainingData, ModelArtifacts, AccuracyMetrics
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

# Model artifacts directory
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# Configuration constants
SUPPORTED_MODEL_TYPES = ["linear_regression", "random_forest", "xgboost", "ensemble"]
DEFAULT_HYPERPARAMETERS = {
    "linear_regression": {},
    "random_forest": {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 5,
        "random_state": 42
    },
    "xgboost": {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "random_state": 42
    }
}

# --- Database Utility Functions ---
def get_db_connection():
    """Get database connection with row factory."""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def create_experiment_record(experiment: ModelExperiment) -> Dict[str, Any]:
    """Create a new model experiment record in the database."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO model_experiments 
            (experiment_id, model_type, hyperparameters, training_data_hash, 
             training_start, training_end, accuracy_score, mape_score, rmse_score, 
             model_artifact_path, experiment_notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            experiment.experiment_id,
            experiment.model_type,
            json.dumps(experiment.hyperparameters),
            experiment.training_data_hash,
            experiment.training_start,
            experiment.training_end,
            experiment.accuracy_score,
            experiment.mape_score,
            experiment.rmse_score,
            experiment.model_artifact_path,
            experiment.experiment_notes
        ))
        
        conn.commit()
        experiment_id = cursor.lastrowid
        conn.close()
        
        return {
            "success": True,
            "message": f"Experiment record created successfully. ID: {experiment_id}",
            "experiment_id": experiment_id
        }
    except sqlite3.Error as e:
        return {
            "success": False,
            "message": f"Error creating experiment record: {e}"
        }


def update_experiment_results(
    experiment_id: str,
    training_end: datetime,
    accuracy_score: float,
    mape_score: Optional[float] = None,
    rmse_score: Optional[float] = None,
    model_artifact_path: Optional[str] = None
) -> Dict[str, Any]:
    """Update experiment results after training completion."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE model_experiments 
            SET training_end = ?, accuracy_score = ?, mape_score = ?, 
                rmse_score = ?, model_artifact_path = ?
            WHERE experiment_id = ?
        """, (
            training_end, accuracy_score, mape_score, rmse_score, 
            model_artifact_path, experiment_id
        ))
        
        conn.commit()
        rows_affected = cursor.rowcount
        conn.close()
        
        if rows_affected > 0:
            return {
                "success": True,
                "message": f"Experiment results updated successfully"
            }
        else:
            return {
                "success": False,
                "message": f"Experiment {experiment_id} not found"
            }
    except sqlite3.Error as e:
        return {
            "success": False,
            "message": f"Error updating experiment results: {e}"
        }


def register_model(model_registry: ModelRegistry) -> Dict[str, Any]:
    """Register a new model in the model registry."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO model_registry 
            (model_id, model_name, version, model_type, status, created_at, 
             performance_metrics, artifact_location, hyperparameters, training_data_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            model_registry.model_id,
            model_registry.model_name,
            model_registry.version,
            model_registry.model_type,
            model_registry.status.value,
            model_registry.created_at,
            json.dumps(model_registry.performance_metrics) if model_registry.performance_metrics else None,
            model_registry.artifact_location,
            json.dumps(model_registry.hyperparameters) if model_registry.hyperparameters else None,
            model_registry.training_data_hash
        ))
        
        conn.commit()
        registry_id = cursor.lastrowid
        conn.close()
        
        return {
            "success": True,
            "message": f"Model registered successfully. Registry ID: {registry_id}",
            "registry_id": registry_id
        }
    except sqlite3.Error as e:
        return {
            "success": False,
            "message": f"Error registering model: {e}"
        }


def get_training_data(days_back: int = 90) -> Optional[TrainingData]:
    """Retrieve training data from sales and inventory tables."""
    try:
        conn = get_db_connection()
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        # Get sales data
        sales_df = pd.read_sql_query("""
            SELECT st.*, p.category, p.subcategory, p.brand
            FROM sales_transactions st
            JOIN products p ON st.product_id = p.id
            WHERE st.transaction_date >= ? AND st.transaction_date <= ?
            ORDER BY st.transaction_date
        """, conn, params=(start_date, end_date))
        
        # Get inventory data
        inventory_df = pd.read_sql_query("""
            SELECT il.*, p.category
            FROM inventory_levels il
            JOIN products p ON il.product_id = p.id
            WHERE il.last_updated >= ?
        """, conn, params=(start_date,))
        
        # Get customer behavior data
        behavior_df = pd.read_sql_query("""
            SELECT * FROM customer_behavior
            WHERE analysis_date >= ?
        """, conn, params=(start_date,))
        
        conn.close()
        
        if sales_df.empty:
            return None
        
        # Create dataset ID
        dataset_id = f"training_data_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
        
        # Calculate quality score based on data completeness
        quality_score = min(1.0, len(sales_df) / 1000)  # Assume 1000+ transactions is good quality
        
        return TrainingData(
            dataset_id=dataset_id,
            date_range=(start_date, end_date),
            sales_data=sales_df.to_dict('records'),
            inventory_data=inventory_df.to_dict('records'),
            customer_behavior=behavior_df.to_dict('records'),
            quality_score=quality_score
        )
        
    except Exception as e:
        logging.error(f"Error retrieving training data: {e}")
        return None


def calculate_data_hash(training_data: TrainingData) -> str:
    """Calculate hash of training data for versioning."""
    data_str = f"{training_data.dataset_id}_{len(training_data.sales_data)}_{training_data.date_range}"
    return hashlib.md5(data_str.encode()).hexdigest()


# --- Model Training Functions ---
def prepare_features(training_data: TrainingData) -> Tuple[pd.DataFrame, pd.Series]:
    """Prepare features and target variables from training data."""
    if not training_data.sales_data:
        raise ValueError("No sales data available for feature preparation")
    
    # Convert to DataFrame
    sales_df = pd.DataFrame(training_data.sales_data)
    
    # Convert transaction_date to datetime
    sales_df['transaction_date'] = pd.to_datetime(sales_df['transaction_date'])
    
    # Create time-based features
    sales_df['year'] = sales_df['transaction_date'].dt.year
    sales_df['month'] = sales_df['transaction_date'].dt.month
    sales_df['day_of_week'] = sales_df['transaction_date'].dt.dayofweek
    sales_df['day_of_year'] = sales_df['transaction_date'].dt.dayofyear
    
    # Aggregate by product and time period (daily)
    sales_df['date'] = sales_df['transaction_date'].dt.date
    
    # Group by product and date to create demand features
    demand_features = sales_df.groupby(['product_id', 'date']).agg({
        'quantity': 'sum',
        'total_amount': 'sum',
        'unit_price': 'mean',
        'year': 'first',
        'month': 'first',
        'day_of_week': 'first',
        'day_of_year': 'first',
        'promotion_applied': 'any'
    }).reset_index()
    
    # Add lag features (previous day demand)
    demand_features = demand_features.sort_values(['product_id', 'date'])
    demand_features['prev_day_quantity'] = demand_features.groupby('product_id')['quantity'].shift(1)
    demand_features['prev_week_quantity'] = demand_features.groupby('product_id')['quantity'].shift(7)
    
    # Fill missing lag features with 0
    demand_features['prev_day_quantity'] = demand_features['prev_day_quantity'].fillna(0)
    demand_features['prev_week_quantity'] = demand_features['prev_week_quantity'].fillna(0)
    
    # Prepare feature matrix (X) and target (y)
    feature_columns = [
        'unit_price', 'year', 'month', 'day_of_week', 'day_of_year',
        'promotion_applied', 'prev_day_quantity', 'prev_week_quantity'
    ]
    
    # Remove rows with missing values
    demand_features = demand_features.dropna(subset=feature_columns + ['quantity'])
    
    X = demand_features[feature_columns]
    y = demand_features['quantity']
    
    return X, y


def train_linear_regression(X: pd.DataFrame, y: pd.Series, hyperparameters: Dict[str, Any]) -> Dict[str, Any]:
    """Train linear regression model."""
    if not ML_AVAILABLE:
        return {"success": False, "message": "Machine learning libraries not available"}
    
    try:
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train model
        model = LinearRegression(**hyperparameters)
        model.fit(X_scaled, y)
        
        # Calculate metrics
        y_pred = model.predict(X_scaled)
        mape = mean_absolute_percentage_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        
        # Calculate accuracy (1 - MAPE)
        accuracy = max(0, 1 - mape)
        
        return {
            "success": True,
            "model": model,
            "scaler": scaler,
            "accuracy_score": accuracy,
            "mape_score": mape,
            "rmse_score": rmse,
            "predictions": y_pred
        }
    except Exception as e:
        return {"success": False, "message": f"Linear regression training failed: {e}"}


def train_random_forest(X: pd.DataFrame, y: pd.Series, hyperparameters: Dict[str, Any]) -> Dict[str, Any]:
    """Train random forest model."""
    if not ML_AVAILABLE:
        return {"success": False, "message": "Machine learning libraries not available"}
    
    try:
        # Train model
        model = RandomForestRegressor(**hyperparameters)
        model.fit(X, y)
        
        # Calculate metrics
        y_pred = model.predict(X)
        mape = mean_absolute_percentage_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        
        # Calculate accuracy (1 - MAPE)
        accuracy = max(0, 1 - mape)
        
        return {
            "success": True,
            "model": model,
            "scaler": None,  # Random Forest doesn't need scaling
            "accuracy_score": accuracy,
            "mape_score": mape,
            "rmse_score": rmse,
            "predictions": y_pred
        }
    except Exception as e:
        return {"success": False, "message": f"Random forest training failed: {e}"}


def train_xgboost(X: pd.DataFrame, y: pd.Series, hyperparameters: Dict[str, Any]) -> Dict[str, Any]:
    """Train XGBoost model."""
    if not ML_AVAILABLE:
        return {"success": False, "message": "Machine learning libraries not available"}
    
    try:
        # Train model
        model = xgb.XGBRegressor(**hyperparameters)
        model.fit(X, y)
        
        # Calculate metrics
        y_pred = model.predict(X)
        mape = mean_absolute_percentage_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        
        # Calculate accuracy (1 - MAPE)
        accuracy = max(0, 1 - mape)
        
        return {
            "success": True,
            "model": model,
            "scaler": None,  # XGBoost doesn't need scaling
            "accuracy_score": accuracy,
            "mape_score": mape,
            "rmse_score": rmse,
            "predictions": y_pred
        }
    except Exception as e:
        return {"success": False, "message": f"XGBoost training failed: {e}"}


def save_model_artifacts(
    model: Any,
    scaler: Optional[Any],
    experiment_id: str,
    model_type: str,
    hyperparameters: Dict[str, Any]
) -> str:
    """Save model artifacts to disk."""
    try:
        # Create model directory
        model_dir = os.path.join(MODELS_DIR, experiment_id)
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(model_dir, "model.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Save scaler if exists
        if scaler is not None:
            scaler_path = os.path.join(model_dir, "scaler.pkl")
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
        
        # Save metadata
        metadata = {
            "model_type": model_type,
            "hyperparameters": hyperparameters,
            "created_at": datetime.now().isoformat(),
            "has_scaler": scaler is not None
        }
        
        metadata_path = os.path.join(model_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return model_dir
    except Exception as e:
        logging.error(f"Error saving model artifacts: {e}")
        raise


# --- Model Artifact and Registry Management Functions ---

def load_model_artifacts(artifact_path: str) -> Dict[str, Any]:
    """Load model artifacts from disk."""
    try:
        if not os.path.exists(artifact_path):
            return {"success": False, "message": f"Artifact path does not exist: {artifact_path}"}
        
        # Load model
        model_path = os.path.join(artifact_path, "model.pkl")
        if not os.path.exists(model_path):
            return {"success": False, "message": "Model file not found"}
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Load metadata
        metadata_path = os.path.join(artifact_path, "metadata.json")
        metadata = {}
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        
        # Load scaler if exists
        scaler = None
        if metadata.get("has_scaler", False):
            scaler_path = os.path.join(artifact_path, "scaler.pkl")
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
        
        return {
            "success": True,
            "model": model,
            "scaler": scaler,
            "metadata": metadata
        }
    except Exception as e:
        return {"success": False, "message": f"Error loading model artifacts: {e}"}


def generate_model_version(model_type: str, existing_versions: List[str] = None) -> str:
    """Generate next version number for a model."""
    if not existing_versions:
        return "1.0.0"
    
    # Parse existing versions and find the highest
    max_major, max_minor, max_patch = 0, 0, 0
    
    for version in existing_versions:
        try:
            parts = version.split('.')
            if len(parts) == 3:
                major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])
                if (major, minor, patch) > (max_major, max_minor, max_patch):
                    max_major, max_minor, max_patch = major, minor, patch
        except ValueError:
            continue
    
    # Increment patch version
    return f"{max_major}.{max_minor}.{max_patch + 1}"


def get_model_registry_entry(model_id: str) -> Optional[Dict[str, Any]]:
    """Get model registry entry by ID."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM model_registry WHERE model_id = ?", (model_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            # Convert row to dict to handle missing columns safely
            row_dict = dict(row)
            return {
                "model_id": row_dict["model_id"],
                "model_name": row_dict["model_name"],
                "version": row_dict["version"],
                "model_type": row_dict["model_type"],
                "status": row_dict["status"],
                "created_at": row_dict["created_at"],
                "deployed_at": row_dict.get("deployed_at"),
                "retired_at": row_dict.get("retired_at"),
                "performance_metrics": json.loads(row_dict["performance_metrics"]) if row_dict["performance_metrics"] else None,
                "artifact_location": row_dict["artifact_location"],
                "hyperparameters": json.loads(row_dict["hyperparameters"]) if row_dict["hyperparameters"] else None,
                "training_data_hash": row_dict.get("training_data_hash")
            }
        return None
    except Exception as e:
        logging.error(f"Error getting model registry entry: {e}")
        return None


def get_existing_model_versions(model_name: str) -> List[str]:
    """Get existing versions for a model name."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT version FROM model_registry WHERE model_name = ?", (model_name,))
        rows = cursor.fetchall()
        conn.close()
        
        return [row["version"] for row in rows]
    except Exception as e:
        logging.error(f"Error getting existing model versions: {e}")
        return []


def create_model_artifacts_package(
    model: Any,
    scaler: Optional[Any],
    model_type: str,
    hyperparameters: Dict[str, Any],
    performance_metrics: Dict[str, Any],
    version: str,
    model_name: str
) -> ModelArtifacts:
    """Create a comprehensive model artifacts package."""
    
    # Generate unique artifact ID
    artifact_id = f"{model_name}_{version}_{uuid.uuid4().hex[:8]}"
    artifact_dir = os.path.join(MODELS_DIR, artifact_id)
    os.makedirs(artifact_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(artifact_dir, "model.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Save scaler if exists
    has_scaler = scaler is not None
    if has_scaler:
        scaler_path = os.path.join(artifact_dir, "scaler.pkl")
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
    
    # Create comprehensive metadata
    metadata = {
        "model_name": model_name,
        "version": version,
        "model_type": model_type,
        "hyperparameters": hyperparameters,
        "performance_metrics": performance_metrics,
        "has_scaler": has_scaler,
        "created_at": datetime.now().isoformat(),
        "artifact_structure": {
            "model.pkl": "Main model object",
            "scaler.pkl": "Feature scaler (if applicable)" if has_scaler else None,
            "metadata.json": "Model metadata and configuration",
            "requirements.txt": "Python dependencies"
        }
    }
    
    # Save metadata
    metadata_path = os.path.join(artifact_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Create requirements.txt for deployment
    requirements_path = os.path.join(artifact_dir, "requirements.txt")
    with open(requirements_path, 'w') as f:
        f.write("scikit-learn>=1.0.0\n")
        f.write("pandas>=1.3.0\n")
        f.write("numpy>=1.21.0\n")
        if model_type == "xgboost":
            f.write("xgboost>=1.5.0\n")
    
    # Create ModelArtifacts object
    artifacts = ModelArtifacts(
        model_id=f"model_{artifact_id}",
        artifact_location=artifact_dir,
        model_type=model_type,
        version=version,
        performance_metrics=performance_metrics,
        hyperparameters=hyperparameters,
        metadata=metadata,
        created_at=datetime.now()
    )
    
    return artifacts


def track_model_performance_history(model_id: str, new_metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Track model performance over time."""
    try:
        # This would typically integrate with a time-series database
        # For now, we'll store in a simple JSON structure
        
        performance_history_path = os.path.join(MODELS_DIR, f"{model_id}_performance_history.json")
        
        # Load existing history
        history = []
        if os.path.exists(performance_history_path):
            with open(performance_history_path, 'r') as f:
                history = json.load(f)
        
        # Add new metrics
        history.append({
            "timestamp": datetime.now().isoformat(),
            "metrics": new_metrics
        })
        
        # Keep only last 100 entries
        history = history[-100:]
        
        # Save updated history
        with open(performance_history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        # Calculate trends
        if len(history) >= 2:
            recent_accuracy = history[-1]["metrics"].get("accuracy_score", 0)
            previous_accuracy = history[-2]["metrics"].get("accuracy_score", 0)
            accuracy_trend = recent_accuracy - previous_accuracy
        else:
            accuracy_trend = 0
        
        return {
            "success": True,
            "history_length": len(history),
            "accuracy_trend": accuracy_trend,
            "latest_metrics": new_metrics
        }
        
    except Exception as e:
        return {"success": False, "message": f"Error tracking performance history: {e}"}


# --- Advanced Hyperparameter Optimization Functions ---
def optimize_hyperparameters_grid_search(
    model_type: str,
    X: pd.DataFrame,
    y: pd.Series,
    cv_folds: int = 3
) -> Dict[str, Any]:
    """Perform grid search hyperparameter optimization."""
    if not ML_AVAILABLE:
        return {"success": False, "message": "Machine learning libraries not available"}
    
    # Define comprehensive parameter grids
    param_grids = {
        "linear_regression": {
            "fit_intercept": [True, False],
            "normalize": [True, False] if hasattr(LinearRegression(), 'normalize') else [True]
        },
        "random_forest": {
            "n_estimators": [50, 100, 200, 300],
            "max_depth": [5, 10, 15, 20, None],
            "min_samples_split": [2, 5, 10, 15],
            "min_samples_leaf": [1, 2, 4, 8],
            "max_features": ["sqrt", "log2", None],
            "bootstrap": [True, False]
        },
        "xgboost": {
            "n_estimators": [50, 100, 200, 300],
            "max_depth": [3, 4, 5, 6, 7, 8],
            "learning_rate": [0.01, 0.05, 0.1, 0.15, 0.2],
            "subsample": [0.8, 0.9, 1.0],
            "colsample_bytree": [0.8, 0.9, 1.0],
            "reg_alpha": [0, 0.1, 0.5, 1.0],
            "reg_lambda": [0, 0.1, 0.5, 1.0]
        }
    }
    
    if model_type not in param_grids:
        return {"success": False, "message": f"Grid search not supported for {model_type}"}
    
    try:
        # Create base model
        if model_type == "linear_regression":
            base_model = LinearRegression()
            # Scale features for linear regression
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            X_to_use = X_scaled
        elif model_type == "random_forest":
            base_model = RandomForestRegressor(random_state=42)
            X_to_use = X
            scaler = None
        elif model_type == "xgboost":
            base_model = xgb.XGBRegressor(random_state=42)
            X_to_use = X
            scaler = None
        
        # Use time series split for cross-validation
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        
        # Perform grid search
        grid_search = GridSearchCV(
            base_model,
            param_grids[model_type],
            cv=tscv,
            scoring='neg_mean_absolute_percentage_error',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_to_use, y)
        
        # Get results
        best_params = grid_search.best_params_
        best_score = -grid_search.best_score_
        
        # Train final model with best parameters
        if model_type == "linear_regression":
            final_model = LinearRegression(**best_params)
            final_model.fit(X_scaled, y)
            y_pred = final_model.predict(X_scaled)
        elif model_type == "random_forest":
            final_model = RandomForestRegressor(**best_params, random_state=42)
            final_model.fit(X, y)
            y_pred = final_model.predict(X)
        elif model_type == "xgboost":
            final_model = xgb.XGBRegressor(**best_params, random_state=42)
            final_model.fit(X, y)
            y_pred = final_model.predict(X)
        
        # Calculate final metrics
        final_mape = mean_absolute_percentage_error(y, y_pred)
        final_rmse = np.sqrt(mean_squared_error(y, y_pred))
        final_accuracy = max(0, 1 - final_mape)
        
        return {
            "success": True,
            "optimization_method": "grid_search",
            "model": final_model,
            "scaler": scaler,
            "best_parameters": best_params,
            "cv_score": best_score,
            "final_accuracy": final_accuracy,
            "final_mape": final_mape,
            "final_rmse": final_rmse,
            "total_combinations": len(grid_search.cv_results_['params'])
        }
        
    except Exception as e:
        return {"success": False, "message": f"Grid search optimization failed: {e}"}


def optimize_hyperparameters_random_search(
    model_type: str,
    X: pd.DataFrame,
    y: pd.Series,
    n_iter: int = 50,
    cv_folds: int = 3
) -> Dict[str, Any]:
    """Perform randomized search hyperparameter optimization."""
    if not ML_AVAILABLE:
        return {"success": False, "message": "Machine learning libraries not available"}
    
    # Define parameter distributions for random search
    param_distributions = {
        "random_forest": {
            "n_estimators": [50, 100, 150, 200, 250, 300],
            "max_depth": [5, 10, 15, 20, 25, None],
            "min_samples_split": [2, 5, 10, 15, 20],
            "min_samples_leaf": [1, 2, 4, 8, 12],
            "max_features": ["sqrt", "log2", None],
            "bootstrap": [True, False]
        },
        "xgboost": {
            "n_estimators": [50, 100, 150, 200, 250, 300],
            "max_depth": [3, 4, 5, 6, 7, 8, 9],
            "learning_rate": [0.01, 0.05, 0.1, 0.15, 0.2, 0.25],
            "subsample": [0.7, 0.8, 0.9, 1.0],
            "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
            "reg_alpha": [0, 0.1, 0.5, 1.0, 2.0],
            "reg_lambda": [0, 0.1, 0.5, 1.0, 2.0]
        }
    }
    
    if model_type not in param_distributions:
        return {"success": False, "message": f"Random search not supported for {model_type}"}
    
    try:
        # Create base model
        if model_type == "random_forest":
            base_model = RandomForestRegressor(random_state=42)
        elif model_type == "xgboost":
            base_model = xgb.XGBRegressor(random_state=42)
        
        # Use time series split for cross-validation
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        
        # Perform randomized search
        random_search = RandomizedSearchCV(
            base_model,
            param_distributions[model_type],
            n_iter=n_iter,
            cv=tscv,
            scoring='neg_mean_absolute_percentage_error',
            n_jobs=-1,
            random_state=42,
            verbose=1
        )
        
        random_search.fit(X, y)
        
        # Get results
        best_params = random_search.best_params_
        best_score = -random_search.best_score_
        
        # Train final model with best parameters
        if model_type == "random_forest":
            final_model = RandomForestRegressor(**best_params, random_state=42)
        elif model_type == "xgboost":
            final_model = xgb.XGBRegressor(**best_params, random_state=42)
        
        final_model.fit(X, y)
        y_pred = final_model.predict(X)
        
        # Calculate final metrics
        final_mape = mean_absolute_percentage_error(y, y_pred)
        final_rmse = np.sqrt(mean_squared_error(y, y_pred))
        final_accuracy = max(0, 1 - final_mape)
        
        return {
            "success": True,
            "optimization_method": "random_search",
            "model": final_model,
            "scaler": None,
            "best_parameters": best_params,
            "cv_score": best_score,
            "final_accuracy": final_accuracy,
            "final_mape": final_mape,
            "final_rmse": final_rmse,
            "iterations": n_iter
        }
        
    except Exception as e:
        return {"success": False, "message": f"Random search optimization failed: {e}"}


def optimize_hyperparameters_bayesian(
    model_type: str,
    X: pd.DataFrame,
    y: pd.Series,
    n_calls: int = 30,
    cv_folds: int = 3
) -> Dict[str, Any]:
    """Perform Bayesian optimization for hyperparameter tuning."""
    if not ML_AVAILABLE or not BAYESIAN_OPT_AVAILABLE:
        return {"success": False, "message": "Bayesian optimization libraries not available"}
    
    # Define search spaces for Bayesian optimization
    search_spaces = {
        "random_forest": {
            "n_estimators": Integer(50, 300),
            "max_depth": Integer(5, 30),
            "min_samples_split": Integer(2, 20),
            "min_samples_leaf": Integer(1, 10),
            "max_features": Categorical(["sqrt", "log2"]),
            "bootstrap": Categorical([True, False])
        },
        "xgboost": {
            "n_estimators": Integer(50, 300),
            "max_depth": Integer(3, 10),
            "learning_rate": Real(0.01, 0.3, prior='log-uniform'),
            "subsample": Real(0.7, 1.0),
            "colsample_bytree": Real(0.7, 1.0),
            "reg_alpha": Real(0.0, 2.0),
            "reg_lambda": Real(0.0, 2.0)
        }
    }
    
    if model_type not in search_spaces:
        return {"success": False, "message": f"Bayesian optimization not supported for {model_type}"}
    
    try:
        # Create base model
        if model_type == "random_forest":
            base_model = RandomForestRegressor(random_state=42)
        elif model_type == "xgboost":
            base_model = xgb.XGBRegressor(random_state=42)
        
        # Use time series split for cross-validation
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        
        # Perform Bayesian optimization
        bayes_search = BayesSearchCV(
            base_model,
            search_spaces[model_type],
            n_iter=n_calls,
            cv=tscv,
            scoring='neg_mean_absolute_percentage_error',
            n_jobs=-1,
            random_state=42,
            verbose=1
        )
        
        bayes_search.fit(X, y)
        
        # Get results
        best_params = bayes_search.best_params_
        best_score = -bayes_search.best_score_
        
        # Train final model with best parameters
        if model_type == "random_forest":
            final_model = RandomForestRegressor(**best_params, random_state=42)
        elif model_type == "xgboost":
            final_model = xgb.XGBRegressor(**best_params, random_state=42)
        
        final_model.fit(X, y)
        y_pred = final_model.predict(X)
        
        # Calculate final metrics
        final_mape = mean_absolute_percentage_error(y, y_pred)
        final_rmse = np.sqrt(mean_squared_error(y, y_pred))
        final_accuracy = max(0, 1 - final_mape)
        
        return {
            "success": True,
            "optimization_method": "bayesian",
            "model": final_model,
            "scaler": None,
            "best_parameters": best_params,
            "cv_score": best_score,
            "final_accuracy": final_accuracy,
            "final_mape": final_mape,
            "final_rmse": final_rmse,
            "function_evaluations": n_calls
        }
        
    except Exception as e:
        return {"success": False, "message": f"Bayesian optimization failed: {e}"}


def perform_cross_validation_analysis(
    model_type: str,
    X: pd.DataFrame,
    y: pd.Series,
    hyperparameters: Dict[str, Any],
    cv_folds: int = 5
) -> Dict[str, Any]:
    """Perform detailed cross-validation analysis for model selection."""
    if not ML_AVAILABLE:
        return {"success": False, "message": "Machine learning libraries not available"}
    
    try:
        # Create model with given hyperparameters
        if model_type == "linear_regression":
            model = LinearRegression(**hyperparameters)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            X_to_use = X_scaled
        elif model_type == "random_forest":
            model = RandomForestRegressor(**hyperparameters, random_state=42)
            X_to_use = X
        elif model_type == "xgboost":
            model = xgb.XGBRegressor(**hyperparameters, random_state=42)
            X_to_use = X
        else:
            return {"success": False, "message": f"Unsupported model type: {model_type}"}
        
        # Perform time series cross-validation
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        
        cv_scores = []
        cv_mape_scores = []
        cv_rmse_scores = []
        fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_to_use)):
            X_train, X_val = X_to_use[train_idx], X_to_use[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Train model on fold
            model.fit(X_train, y_train)
            
            # Predict on validation set
            y_pred = model.predict(X_val)
            
            # Calculate metrics
            mape = mean_absolute_percentage_error(y_val, y_pred)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            accuracy = max(0, 1 - mape)
            
            cv_scores.append(accuracy)
            cv_mape_scores.append(mape)
            cv_rmse_scores.append(rmse)
            
            fold_results.append({
                "fold": fold + 1,
                "train_size": len(train_idx),
                "val_size": len(val_idx),
                "accuracy": accuracy,
                "mape": mape,
                "rmse": rmse
            })
        
        # Calculate summary statistics
        cv_mean = np.mean(cv_scores)
        cv_std = np.std(cv_scores)
        cv_min = np.min(cv_scores)
        cv_max = np.max(cv_scores)
        
        return {
            "success": True,
            "model_type": model_type,
            "hyperparameters": hyperparameters,
            "cv_folds": cv_folds,
            "cv_mean_accuracy": cv_mean,
            "cv_std_accuracy": cv_std,
            "cv_min_accuracy": cv_min,
            "cv_max_accuracy": cv_max,
            "cv_mean_mape": np.mean(cv_mape_scores),
            "cv_mean_rmse": np.mean(cv_rmse_scores),
            "fold_results": fold_results,
            "stability_score": 1 - (cv_std / cv_mean) if cv_mean > 0 else 0
        }
        
    except Exception as e:
        return {"success": False, "message": f"Cross-validation analysis failed: {e}"}


def create_ensemble_with_stacking(
    base_models: List[Dict[str, Any]],
    X: pd.DataFrame,
    y: pd.Series,
    meta_model_type: str = "linear_regression"
) -> Dict[str, Any]:
    """Create ensemble model using stacking approach."""
    if not ML_AVAILABLE or len(base_models) < 2:
        return {"success": False, "message": "Insufficient models or ML libraries not available"}
    
    try:
        # Prepare base model predictions for meta-model training
        base_predictions = []
        trained_base_models = []
        
        # Use time series split for stacking
        tscv = TimeSeriesSplit(n_splits=3)
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            fold_predictions = []
            
            for base_model_info in base_models:
                model_type = base_model_info["model_type"]
                hyperparameters = base_model_info.get("hyperparameters", {})
                
                # Train base model
                if model_type == "linear_regression":
                    model = LinearRegression(**hyperparameters)
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_val_scaled = scaler.transform(X_val)
                    model.fit(X_train_scaled, y_train)
                    pred = model.predict(X_val_scaled)
                elif model_type == "random_forest":
                    model = RandomForestRegressor(**hyperparameters, random_state=42)
                    model.fit(X_train, y_train)
                    pred = model.predict(X_val)
                elif model_type == "xgboost":
                    model = xgb.XGBRegressor(**hyperparameters, random_state=42)
                    model.fit(X_train, y_train)
                    pred = model.predict(X_val)
                
                fold_predictions.append(pred)
            
            base_predictions.append(np.column_stack(fold_predictions))
        
        # Combine all fold predictions
        all_base_predictions = np.vstack(base_predictions)
        all_targets = np.concatenate([y.iloc[val_idx] for _, val_idx in tscv.split(X)])
        
        # Train meta-model
        if meta_model_type == "linear_regression":
            meta_model = LinearRegression()
        elif meta_model_type == "random_forest":
            meta_model = RandomForestRegressor(n_estimators=50, random_state=42)
        else:
            meta_model = LinearRegression()  # Default fallback
        
        meta_model.fit(all_base_predictions, all_targets)
        
        # Train final base models on full dataset
        final_base_models = []
        for base_model_info in base_models:
            model_type = base_model_info["model_type"]
            hyperparameters = base_model_info.get("hyperparameters", {})
            
            if model_type == "linear_regression":
                model = LinearRegression(**hyperparameters)
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                model.fit(X_scaled, y)
                final_base_models.append({"model": model, "scaler": scaler, "type": model_type})
            elif model_type == "random_forest":
                model = RandomForestRegressor(**hyperparameters, random_state=42)
                model.fit(X, y)
                final_base_models.append({"model": model, "scaler": None, "type": model_type})
            elif model_type == "xgboost":
                model = xgb.XGBRegressor(**hyperparameters, random_state=42)
                model.fit(X, y)
                final_base_models.append({"model": model, "scaler": None, "type": model_type})
        
        # Evaluate ensemble performance
        final_base_preds = []
        for base_model_info in final_base_models:
            model = base_model_info["model"]
            scaler = base_model_info["scaler"]
            
            if scaler is not None:
                X_scaled = scaler.transform(X)
                pred = model.predict(X_scaled)
            else:
                pred = model.predict(X)
            
            final_base_preds.append(pred)
        
        final_base_preds = np.column_stack(final_base_preds)
        ensemble_pred = meta_model.predict(final_base_preds)
        
        # Calculate ensemble metrics
        ensemble_mape = mean_absolute_percentage_error(y, ensemble_pred)
        ensemble_rmse = np.sqrt(mean_squared_error(y, ensemble_pred))
        ensemble_accuracy = max(0, 1 - ensemble_mape)
        
        return {
            "success": True,
            "ensemble_type": "stacking",
            "meta_model_type": meta_model_type,
            "base_models": final_base_models,
            "meta_model": meta_model,
            "ensemble_accuracy": ensemble_accuracy,
            "ensemble_mape": ensemble_mape,
            "ensemble_rmse": ensemble_rmse,
            "num_base_models": len(final_base_models)
        }
        
    except Exception as e:
        return {"success": False, "message": f"Stacking ensemble creation failed: {e}"}


# --- MCP Server Implementation ---
server = Server("forecasting-model-mcp")

@server.list_tools()
async def handle_list_tools() -> List[mcp_types.Tool]:
    """List available forecasting model tools."""
    return [
        mcp_types.Tool(
            name="train_forecasting_models",
            description="Train multiple forecasting models using collected data",
            inputSchema={
                "type": "object",
                "properties": {
                    "days_back": {
                        "type": "integer",
                        "description": "Number of days of historical data to use for training",
                        "default": 90
                    },
                    "model_types": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of model types to train",
                        "default": ["linear_regression", "random_forest", "xgboost"]
                    }
                }
            }
        ),
        mcp_types.Tool(
            name="optimize_hyperparameters",
            description="Optimize hyperparameters for a specific model type using various methods",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_type": {
                        "type": "string",
                        "description": "Type of model to optimize",
                        "enum": SUPPORTED_MODEL_TYPES
                    },
                    "optimization_method": {
                        "type": "string",
                        "description": "Hyperparameter optimization method",
                        "enum": ["grid_search", "random_search", "bayesian"],
                        "default": "grid_search"
                    },
                    "days_back": {
                        "type": "integer",
                        "description": "Number of days of historical data to use",
                        "default": 90
                    },
                    "n_iter": {
                        "type": "integer",
                        "description": "Number of iterations for random/bayesian search",
                        "default": 50
                    },
                    "cv_folds": {
                        "type": "integer",
                        "description": "Number of cross-validation folds",
                        "default": 3
                    }
                },
                "required": ["model_type"]
            }
        ),
        mcp_types.Tool(
            name="perform_cross_validation",
            description="Perform detailed cross-validation analysis for model selection",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_type": {
                        "type": "string",
                        "description": "Type of model to validate",
                        "enum": SUPPORTED_MODEL_TYPES
                    },
                    "hyperparameters": {
                        "type": "object",
                        "description": "Model hyperparameters to validate"
                    },
                    "days_back": {
                        "type": "integer",
                        "description": "Number of days of historical data to use",
                        "default": 90
                    },
                    "cv_folds": {
                        "type": "integer",
                        "description": "Number of cross-validation folds",
                        "default": 5
                    }
                },
                "required": ["model_type", "hyperparameters"]
            }
        ),
        mcp_types.Tool(
            name="create_ensemble_model",
            description="Create ensemble model from multiple base models using various methods",
            inputSchema={
                "type": "object",
                "properties": {
                    "base_model_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of base model experiment IDs to combine"
                    },
                    "ensemble_method": {
                        "type": "string",
                        "description": "Ensemble combination method",
                        "enum": ["average", "weighted_average", "stacking"],
                        "default": "average"
                    },
                    "meta_model_type": {
                        "type": "string",
                        "description": "Meta-model type for stacking (only used with stacking method)",
                        "enum": ["linear_regression", "random_forest"],
                        "default": "linear_regression"
                    },
                    "days_back": {
                        "type": "integer",
                        "description": "Number of days of historical data to use for stacking",
                        "default": 90
                    }
                },
                "required": ["base_model_ids"]
            }
        ),
        mcp_types.Tool(
            name="generate_model_artifacts",
            description="Generate deployment artifacts for a trained model",
            inputSchema={
                "type": "object",
                "properties": {
                    "experiment_id": {
                        "type": "string",
                        "description": "Experiment ID of the model to generate artifacts for"
                    },
                    "version": {
                        "type": "string",
                        "description": "Model version (defaults to auto-generated)",
                        "default": "auto"
                    },
                    "model_name": {
                        "type": "string",
                        "description": "Custom model name (optional)"
                    }
                },
                "required": ["experiment_id"]
            }
        ),
        mcp_types.Tool(
            name="list_model_registry",
            description="List models in the registry with filtering options",
            inputSchema={
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "description": "Filter by model status",
                        "enum": ["TRAINING", "VALIDATION", "PRODUCTION", "RETIRED"]
                    },
                    "model_type": {
                        "type": "string",
                        "description": "Filter by model type"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of models to return",
                        "default": 50
                    }
                }
            }
        ),
        mcp_types.Tool(
            name="compare_model_performance",
            description="Compare performance metrics between multiple models",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of model IDs to compare"
                    },
                    "metrics": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Metrics to compare",
                        "default": ["accuracy_score", "mape_score", "rmse_score"]
                    }
                },
                "required": ["model_ids"]
            }
        ),
        mcp_types.Tool(
            name="update_model_status",
            description="Update model status in the registry",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": {
                        "type": "string",
                        "description": "Model ID to update"
                    },
                    "new_status": {
                        "type": "string",
                        "description": "New status for the model",
                        "enum": ["TRAINING", "VALIDATION", "PRODUCTION", "RETIRED"]
                    },
                    "notes": {
                        "type": "string",
                        "description": "Optional notes about the status change"
                    }
                },
                "required": ["model_id", "new_status"]
            }
        )
    ]


@server.call_tool()
async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> List[mcp_types.TextContent]:
    """Handle tool calls for forecasting model operations."""
    
    if name == "train_forecasting_models":
        return await train_forecasting_models_tool(arguments)
    elif name == "optimize_hyperparameters":
        return await optimize_hyperparameters_tool(arguments)
    elif name == "perform_cross_validation":
        return await perform_cross_validation_tool(arguments)
    elif name == "create_ensemble_model":
        return await create_ensemble_model_tool(arguments)
    elif name == "generate_model_artifacts":
        return await generate_model_artifacts_tool(arguments)
    elif name == "list_model_registry":
        return await list_model_registry_tool(arguments)
    elif name == "compare_model_performance":
        return await compare_model_performance_tool(arguments)
    elif name == "update_model_status":
        return await update_model_status_tool(arguments)
    else:
        return [mcp_types.TextContent(
            type="text",
            text=f"Unknown tool: {name}"
        )]


async def train_forecasting_models_tool(arguments: Dict[str, Any]) -> List[mcp_types.TextContent]:
    """Train multiple forecasting models using collected data."""
    try:
        days_back = arguments.get("days_back", 90)
        model_types = arguments.get("model_types", ["linear_regression", "random_forest", "xgboost"])
        
        # Get training data
        training_data = get_training_data(days_back)
        if not training_data:
            return [mcp_types.TextContent(
                type="text",
                text="Error: No training data available for the specified period"
            )]
        
        # Prepare features
        try:
            X, y = prepare_features(training_data)
        except Exception as e:
            return [mcp_types.TextContent(
                type="text",
                text=f"Error preparing features: {e}"
            )]
        
        # Calculate data hash
        data_hash = calculate_data_hash(training_data)
        
        results = []
        experiments = []
        
        for model_type in model_types:
            if model_type not in SUPPORTED_MODEL_TYPES:
                results.append(f"Skipping unsupported model type: {model_type}")
                continue
            
            # Create experiment record
            experiment_id = f"{model_type}_{uuid.uuid4().hex[:8]}"
            hyperparameters = DEFAULT_HYPERPARAMETERS.get(model_type, {})
            
            experiment = ModelExperiment(
                experiment_id=experiment_id,
                model_type=model_type,
                training_start=datetime.now(),
                hyperparameters=hyperparameters,
                training_data_hash=data_hash
            )
            
            # Create database record
            create_result = create_experiment_record(experiment)
            if not create_result["success"]:
                results.append(f"Failed to create experiment record for {model_type}: {create_result['message']}")
                continue
            
            # Train model
            training_result = None
            if model_type == "linear_regression":
                training_result = train_linear_regression(X, y, hyperparameters)
            elif model_type == "random_forest":
                training_result = train_random_forest(X, y, hyperparameters)
            elif model_type == "xgboost":
                training_result = train_xgboost(X, y, hyperparameters)
            
            if not training_result or not training_result["success"]:
                error_msg = training_result.get("message", "Unknown error") if training_result else "Training failed"
                results.append(f"Training failed for {model_type}: {error_msg}")
                continue
            
            # Save model artifacts
            try:
                artifact_path = save_model_artifacts(
                    training_result["model"],
                    training_result.get("scaler"),
                    experiment_id,
                    model_type,
                    hyperparameters
                )
            except Exception as e:
                results.append(f"Failed to save artifacts for {model_type}: {e}")
                continue
            
            # Update experiment results
            update_result = update_experiment_results(
                experiment_id,
                datetime.now(),
                training_result["accuracy_score"],
                training_result["mape_score"],
                training_result["rmse_score"],
                artifact_path
            )
            
            if update_result["success"]:
                experiments.append({
                    "experiment_id": experiment_id,
                    "model_type": model_type,
                    "accuracy_score": training_result["accuracy_score"],
                    "mape_score": training_result["mape_score"],
                    "rmse_score": training_result["rmse_score"]
                })
                results.append(f"Successfully trained {model_type} - Accuracy: {training_result['accuracy_score']:.3f}")
            else:
                results.append(f"Failed to update results for {model_type}: {update_result['message']}")
        
        # Summary
        summary = {
            "total_models_trained": len(experiments),
            "training_data_period": f"{training_data.date_range[0].strftime('%Y-%m-%d')} to {training_data.date_range[1].strftime('%Y-%m-%d')}",
            "training_samples": len(X),
            "experiments": experiments,
            "results": results
        }
        
        return [mcp_types.TextContent(
            type="text",
            text=json.dumps(summary, indent=2)
        )]
        
    except Exception as e:
        return [mcp_types.TextContent(
            type="text",
            text=f"Error in model training: {e}"
        )]


async def optimize_hyperparameters_tool(arguments: Dict[str, Any]) -> List[mcp_types.TextContent]:
    """Optimize hyperparameters for a specific model type using various methods."""
    try:
        model_type = arguments["model_type"]
        optimization_method = arguments.get("optimization_method", "grid_search")
        days_back = arguments.get("days_back", 90)
        n_iter = arguments.get("n_iter", 50)
        cv_folds = arguments.get("cv_folds", 3)
        
        if model_type not in SUPPORTED_MODEL_TYPES:
            return [mcp_types.TextContent(
                type="text",
                text=f"Unsupported model type: {model_type}"
            )]
        
        # Get training data
        training_data = get_training_data(days_back)
        if not training_data:
            return [mcp_types.TextContent(
                type="text",
                text="Error: No training data available for the specified period"
            )]
        
        # Prepare features
        X, y = prepare_features(training_data)
        
        # Perform optimization based on selected method
        optimization_result = None
        
        if optimization_method == "grid_search":
            optimization_result = optimize_hyperparameters_grid_search(model_type, X, y, cv_folds)
        elif optimization_method == "random_search":
            optimization_result = optimize_hyperparameters_random_search(model_type, X, y, n_iter, cv_folds)
        elif optimization_method == "bayesian":
            optimization_result = optimize_hyperparameters_bayesian(model_type, X, y, n_iter, cv_folds)
        else:
            return [mcp_types.TextContent(
                type="text",
                text=f"Unsupported optimization method: {optimization_method}"
            )]
        
        if not optimization_result or not optimization_result["success"]:
            error_msg = optimization_result.get("message", "Unknown error") if optimization_result else "Optimization failed"
            return [mcp_types.TextContent(
                type="text",
                text=f"Hyperparameter optimization failed: {error_msg}"
            )]
        
        # Create experiment record
        experiment_id = f"{model_type}_{optimization_method}_{uuid.uuid4().hex[:8]}"
        
        # Save optimized model
        artifact_path = save_model_artifacts(
            optimization_result["model"],
            optimization_result.get("scaler"),
            experiment_id,
            model_type,
            optimization_result["best_parameters"]
        )
        
        # Create experiment record
        data_hash = calculate_data_hash(training_data)
        experiment = ModelExperiment(
            experiment_id=experiment_id,
            model_type=model_type,
            training_start=datetime.now(),
            training_end=datetime.now(),
            hyperparameters=optimization_result["best_parameters"],
            training_data_hash=data_hash,
            accuracy_score=optimization_result["final_accuracy"],
            mape_score=optimization_result["final_mape"],
            rmse_score=optimization_result["final_rmse"],
            model_artifact_path=artifact_path,
            experiment_notes=f"Hyperparameter optimized model using {optimization_method}"
        )
        
        create_experiment_record(experiment)
        
        result = {
            "success": True,
            "experiment_id": experiment_id,
            "model_type": model_type,
            "optimization_method": optimization_method,
            "best_parameters": optimization_result["best_parameters"],
            "cross_validation_score": optimization_result["cv_score"],
            "final_accuracy": optimization_result["final_accuracy"],
            "final_mape": optimization_result["final_mape"],
            "final_rmse": optimization_result["final_rmse"],
            "optimization_details": {
                key: value for key, value in optimization_result.items() 
                if key not in ["model", "scaler", "success"]
            }
        }
        
        return [mcp_types.TextContent(
            type="text",
            text=json.dumps(result, indent=2)
        )]
        
    except Exception as e:
        return [mcp_types.TextContent(
            type="text",
            text=f"Error in hyperparameter optimization: {e}"
        )]


async def perform_cross_validation_tool(arguments: Dict[str, Any]) -> List[mcp_types.TextContent]:
    """Perform detailed cross-validation analysis for model selection."""
    try:
        model_type = arguments["model_type"]
        hyperparameters = arguments["hyperparameters"]
        days_back = arguments.get("days_back", 90)
        cv_folds = arguments.get("cv_folds", 5)
        
        if model_type not in SUPPORTED_MODEL_TYPES:
            return [mcp_types.TextContent(
                type="text",
                text=f"Unsupported model type: {model_type}"
            )]
        
        # Get training data
        training_data = get_training_data(days_back)
        if not training_data:
            return [mcp_types.TextContent(
                type="text",
                text="Error: No training data available for the specified period"
            )]
        
        # Prepare features
        X, y = prepare_features(training_data)
        
        # Perform cross-validation analysis
        cv_result = perform_cross_validation_analysis(model_type, X, y, hyperparameters, cv_folds)
        
        if not cv_result["success"]:
            return [mcp_types.TextContent(
                type="text",
                text=f"Cross-validation analysis failed: {cv_result['message']}"
            )]
        
        return [mcp_types.TextContent(
            type="text",
            text=json.dumps(cv_result, indent=2)
        )]
        
    except Exception as e:
        return [mcp_types.TextContent(
            type="text",
            text=f"Error in cross-validation analysis: {e}"
        )]


async def create_ensemble_model_tool(arguments: Dict[str, Any]) -> List[mcp_types.TextContent]:
    """Create ensemble model from multiple base models."""
    try:
        base_model_ids = arguments["base_model_ids"]
        ensemble_method = arguments.get("ensemble_method", "average")
        
        if len(base_model_ids) < 2:
            return [mcp_types.TextContent(
                type="text",
                text="Error: At least 2 base models required for ensemble"
            )]
        
        # Load base models and their performance
        base_models = []
        model_performances = []
        
        for model_id in base_model_ids:
            try:
                # Load model from experiment record
                conn = get_db_connection()
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM model_experiments WHERE experiment_id = ?
                """, (model_id,))
                row = cursor.fetchone()
                conn.close()
                
                if not row:
                    return [mcp_types.TextContent(
                        type="text",
                        text=f"Error: Model {model_id} not found in experiments"
                    )]
                
                # Load model artifacts
                model_dir = row["model_artifact_path"]
                if not model_dir or not os.path.exists(model_dir):
                    return [mcp_types.TextContent(
                        type="text",
                        text=f"Error: Model artifacts not found for {model_id}"
                    )]
                
                # Load model and metadata
                model_path = os.path.join(model_dir, "model.pkl")
                metadata_path = os.path.join(model_dir, "metadata.json")
                
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                scaler = None
                if metadata.get("has_scaler", False):
                    scaler_path = os.path.join(model_dir, "scaler.pkl")
                    with open(scaler_path, 'rb') as f:
                        scaler = pickle.load(f)
                
                base_models.append({
                    "model_type": row["model_type"],
                    "model": model,
                    "scaler": scaler,
                    "hyperparameters": json.loads(row["hyperparameters"]) if row["hyperparameters"] else {}
                })
                
                model_performances.append({
                    "model_id": model_id,
                    "accuracy": row["accuracy_score"],
                    "mape": row["mape_score"],
                    "rmse": row["rmse_score"]
                })
                
            except Exception as e:
                return [mcp_types.TextContent(
                    type="text",
                    text=f"Error loading model {model_id}: {e}"
                )]
        
        # Get training data for ensemble creation
        days_back = arguments.get("days_back", 90)
        training_data = get_training_data(days_back)
        if not training_data:
            return [mcp_types.TextContent(
                type="text",
                text="Error: No training data available for ensemble creation"
            )]
        
        X, y = prepare_features(training_data)
        
        # Create ensemble based on method
        ensemble_result = None
        
        if ensemble_method == "stacking":
            meta_model_type = arguments.get("meta_model_type", "linear_regression")
            ensemble_result = create_ensemble_with_stacking(base_models, X, y, meta_model_type)
        else:
            # Simple averaging or weighted averaging
            ensemble_predictions = []
            
            for base_model_info in base_models:
                model = base_model_info["model"]
                scaler = base_model_info["scaler"]
                
                if scaler is not None:
                    X_scaled = scaler.transform(X)
                    pred = model.predict(X_scaled)
                else:
                    pred = model.predict(X)
                
                ensemble_predictions.append(pred)
            
            ensemble_predictions = np.column_stack(ensemble_predictions)
            
            if ensemble_method == "average":
                final_pred = np.mean(ensemble_predictions, axis=1)
            elif ensemble_method == "weighted_average":
                # Weight by accuracy
                weights = np.array([perf["accuracy"] for perf in model_performances])
                weights = weights / np.sum(weights)  # Normalize
                final_pred = np.average(ensemble_predictions, axis=1, weights=weights)
            
            # Calculate ensemble metrics
            ensemble_mape = mean_absolute_percentage_error(y, final_pred)
            ensemble_rmse = np.sqrt(mean_squared_error(y, final_pred))
            ensemble_accuracy = max(0, 1 - ensemble_mape)
            
            ensemble_result = {
                "success": True,
                "ensemble_type": ensemble_method,
                "ensemble_accuracy": ensemble_accuracy,
                "ensemble_mape": ensemble_mape,
                "ensemble_rmse": ensemble_rmse,
                "num_base_models": len(base_models),
                "base_model_weights": weights.tolist() if ensemble_method == "weighted_average" else None
            }
        
        if not ensemble_result or not ensemble_result["success"]:
            return [mcp_types.TextContent(
                type="text",
                text=f"Ensemble creation failed: {ensemble_result.get('message', 'Unknown error')}"
            )]
        
        # Create ensemble experiment record
        ensemble_id = f"ensemble_{ensemble_method}_{uuid.uuid4().hex[:8]}"
        
        ensemble_experiment = ModelExperiment(
            experiment_id=ensemble_id,
            model_type="ensemble",
            training_start=datetime.now(),
            training_end=datetime.now(),
            hyperparameters={
                "ensemble_method": ensemble_method,
                "base_models": base_model_ids,
                "meta_model_type": arguments.get("meta_model_type") if ensemble_method == "stacking" else None
            },
            training_data_hash=calculate_data_hash(training_data),
            accuracy_score=ensemble_result["ensemble_accuracy"],
            mape_score=ensemble_result["ensemble_mape"],
            rmse_score=ensemble_result["ensemble_rmse"],
            experiment_notes=f"Ensemble model using {ensemble_method} method"
        )
        
        create_experiment_record(ensemble_experiment)
        
        result = {
            "success": True,
            "ensemble_id": ensemble_id,
            "ensemble_method": ensemble_method,
            "base_models": base_model_ids,
            "performance": {
                "accuracy": ensemble_result["ensemble_accuracy"],
                "mape": ensemble_result["ensemble_mape"],
                "rmse": ensemble_result["ensemble_rmse"]
            },
            "ensemble_details": ensemble_result
        }
        
        return [mcp_types.TextContent(
            type="text",
            text=json.dumps(result, indent=2)
        )]
        
    except Exception as e:
        return [mcp_types.TextContent(
            type="text",
            text=f"Error creating ensemble model: {e}"
        )]


async def list_model_registry_tool(arguments: Dict[str, Any]) -> List[mcp_types.TextContent]:
    """List models in the registry with filtering options."""
    try:
        status_filter = arguments.get("status")
        model_type_filter = arguments.get("model_type")
        limit = arguments.get("limit", 50)
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Build query with filters
        query = "SELECT * FROM model_registry WHERE 1=1"
        params = []
        
        if status_filter:
            query += " AND status = ?"
            params.append(status_filter)
        
        if model_type_filter:
            query += " AND model_type = ?"
            params.append(model_type_filter)
        
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        models = []
        for row in rows:
            model_info = {
                "model_id": row["model_id"],
                "model_name": row["model_name"],
                "version": row["version"],
                "model_type": row["model_type"],
                "status": row["status"],
                "created_at": row["created_at"],
                "deployed_at": row["deployed_at"],
                "performance_metrics": json.loads(row["performance_metrics"]) if row["performance_metrics"] else None,
                "artifact_location": row["artifact_location"]
            }
            models.append(model_info)
        
        result = {
            "total_models": len(models),
            "filters_applied": {
                "status": status_filter,
                "model_type": model_type_filter,
                "limit": limit
            },
            "models": models
        }
        
        return [mcp_types.TextContent(
            type="text",
            text=json.dumps(result, indent=2)
        )]
        
    except Exception as e:
        return [mcp_types.TextContent(
            type="text",
            text=f"Error listing model registry: {e}"
        )]


async def compare_model_performance_tool(arguments: Dict[str, Any]) -> List[mcp_types.TextContent]:
    """Compare performance metrics between multiple models."""
    try:
        model_ids = arguments["model_ids"]
        metrics = arguments.get("metrics", ["accuracy_score", "mape_score", "rmse_score"])
        
        if len(model_ids) < 2:
            return [mcp_types.TextContent(
                type="text",
                text="Error: At least 2 models required for comparison"
            )]
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        model_data = []
        for model_id in model_ids:
            # Try to find in model registry first
            cursor.execute("SELECT * FROM model_registry WHERE model_id = ?", (model_id,))
            registry_row = cursor.fetchone()
            
            if registry_row:
                performance_metrics = json.loads(registry_row["performance_metrics"]) if registry_row["performance_metrics"] else {}
                model_info = {
                    "model_id": model_id,
                    "model_name": registry_row["model_name"],
                    "model_type": registry_row["model_type"],
                    "version": registry_row["version"],
                    "status": registry_row["status"],
                    "created_at": registry_row["created_at"],
                    "metrics": performance_metrics
                }
            else:
                # Try to find in experiments
                cursor.execute("SELECT * FROM model_experiments WHERE experiment_id = ?", (model_id,))
                exp_row = cursor.fetchone()
                
                if exp_row:
                    model_info = {
                        "model_id": model_id,
                        "model_name": f"{exp_row['model_type']}_experiment",
                        "model_type": exp_row["model_type"],
                        "version": "experimental",
                        "status": "TRAINING",
                        "created_at": exp_row["training_start"],
                        "metrics": {
                            "accuracy_score": exp_row["accuracy_score"],
                            "mape_score": exp_row["mape_score"],
                            "rmse_score": exp_row["rmse_score"]
                        }
                    }
                else:
                    return [mcp_types.TextContent(
                        type="text",
                        text=f"Error: Model {model_id} not found in registry or experiments"
                    )]
            
            model_data.append(model_info)
        
        conn.close()
        
        # Create comparison table
        comparison = {
            "models_compared": len(model_data),
            "comparison_metrics": metrics,
            "models": model_data,
            "metric_comparison": {}
        }
        
        # Compare each metric
        for metric in metrics:
            metric_values = []
            for model in model_data:
                value = model["metrics"].get(metric)
                metric_values.append({
                    "model_id": model["model_id"],
                    "model_name": model["model_name"],
                    "value": value
                })
            
            # Sort by metric value (higher is better for accuracy, lower is better for MAPE/RMSE)
            if metric == "accuracy_score":
                metric_values.sort(key=lambda x: x["value"] if x["value"] is not None else -1, reverse=True)
            else:
                metric_values.sort(key=lambda x: x["value"] if x["value"] is not None else float('inf'))
            
            comparison["metric_comparison"][metric] = {
                "best_model": metric_values[0] if metric_values else None,
                "worst_model": metric_values[-1] if metric_values else None,
                "all_values": metric_values
            }
        
        # Overall ranking (based on accuracy)
        if "accuracy_score" in metrics:
            accuracy_ranking = comparison["metric_comparison"]["accuracy_score"]["all_values"]
            comparison["overall_ranking"] = accuracy_ranking
        
        return [mcp_types.TextContent(
            type="text",
            text=json.dumps(comparison, indent=2)
        )]
        
    except Exception as e:
        return [mcp_types.TextContent(
            type="text",
            text=f"Error comparing model performance: {e}"
        )]


async def update_model_status_tool(arguments: Dict[str, Any]) -> List[mcp_types.TextContent]:
    """Update model status in the registry."""
    try:
        model_id = arguments["model_id"]
        new_status = arguments["new_status"]
        notes = arguments.get("notes", "")
        
        # Validate status
        valid_statuses = ["TRAINING", "VALIDATION", "PRODUCTION", "RETIRED"]
        if new_status not in valid_statuses:
            return [mcp_types.TextContent(
                type="text",
                text=f"Error: Invalid status. Must be one of: {valid_statuses}"
            )]
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check if model exists
        cursor.execute("SELECT * FROM model_registry WHERE model_id = ?", (model_id,))
        row = cursor.fetchone()
        
        if not row:
            conn.close()
            return [mcp_types.TextContent(
                type="text",
                text=f"Error: Model {model_id} not found in registry"
            )]
        
        old_status = row["status"]
        
        # Update status
        update_fields = ["status = ?"]
        params = [new_status]
        
        # Set deployed_at if moving to PRODUCTION
        if new_status == "PRODUCTION" and old_status != "PRODUCTION":
            update_fields.append("deployed_at = ?")
            params.append(datetime.now())
        
        # Set retired_at if moving to RETIRED
        if new_status == "RETIRED" and old_status != "RETIRED":
            update_fields.append("retired_at = ?")
            params.append(datetime.now())
        
        params.append(model_id)
        
        cursor.execute(f"""
            UPDATE model_registry 
            SET {', '.join(update_fields)}
            WHERE model_id = ?
        """, params)
        
        conn.commit()
        rows_affected = cursor.rowcount
        conn.close()
        
        if rows_affected > 0:
            result = {
                "success": True,
                "model_id": model_id,
                "old_status": old_status,
                "new_status": new_status,
                "updated_at": datetime.now().isoformat(),
                "notes": notes
            }
        else:
            result = {
                "success": False,
                "message": f"Failed to update status for model {model_id}"
            }
        
        return [mcp_types.TextContent(
            type="text",
            text=json.dumps(result, indent=2)
        )]
        
    except Exception as e:
        return [mcp_types.TextContent(
            type="text",
            text=f"Error updating model status: {e}"
        )]


async def generate_model_artifacts_tool(arguments: Dict[str, Any]) -> List[mcp_types.TextContent]:
    """Generate deployment artifacts for a trained model."""
    try:
        experiment_id = arguments["experiment_id"]
        
        # Get experiment details from database
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM model_experiments
            WHERE experiment_id = ?
        """, (experiment_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return [mcp_types.TextContent(
                type="text",
                text=f"Experiment {experiment_id} not found"
            )]
        
        # Check if model artifacts exist
        model_dir = os.path.join(MODELS_DIR, experiment_id)
        if not os.path.exists(model_dir):
            return [mcp_types.TextContent(
                type="text",
                text=f"Model artifacts not found for experiment {experiment_id}"
            )]
        
        # Create model registry entry
        model_id = f"model_{experiment_id}"
        version = "1.0.0"
        
        performance_metrics = {
            "accuracy_score": row["accuracy_score"],
            "mape_score": row["mape_score"],
            "rmse_score": row["rmse_score"]
        }
        
        hyperparameters = json.loads(row["hyperparameters"]) if row["hyperparameters"] else {}
        
        model_registry = ModelRegistry(
            model_id=model_id,
            model_name=f"{row['model_type']}_demand_forecast",
            version=version,
            model_type=row["model_type"],
            status=ModelStatus.VALIDATION,
            created_at=datetime.now(),
            performance_metrics=performance_metrics,
            artifact_location=model_dir,
            hyperparameters=hyperparameters,
            training_data_hash=row["training_data_hash"]
        )
        
        # Register model
        register_result = register_model(model_registry)
        
        if register_result["success"]:
            # Create model artifacts object
            artifacts = ModelArtifacts(
                model_id=model_id,
                artifact_location=model_dir,
                model_type=row["model_type"],
                version=version,
                performance_metrics=performance_metrics,
                hyperparameters=hyperparameters,
                metadata={
                    "experiment_id": experiment_id,
                    "training_start": row["training_start"],
                    "training_end": row["training_end"],
                    "training_data_hash": row["training_data_hash"]
                },
                created_at=datetime.now()
            )
            
            result = {
                "success": True,
                "model_id": model_id,
                "version": version,
                "artifact_location": model_dir,
                "performance_metrics": performance_metrics,
                "registry_id": register_result["registry_id"],
                "status": "VALIDATION"
            }
        else:
            result = {
                "success": False,
                "message": f"Failed to register model: {register_result['message']}"
            }
        
        return [mcp_types.TextContent(
            type="text",
            text=json.dumps(result, indent=2)
        )]
        
    except Exception as e:
        return [mcp_types.TextContent(
            type="text",
            text=f"Error generating model artifacts: {e}"
        )]


async def generate_model_artifacts_tool(arguments: Dict[str, Any]) -> List[mcp_types.TextContent]:
    """Generate deployment artifacts for a trained model."""
    try:
        experiment_id = arguments["experiment_id"]
        version_arg = arguments.get("version", "auto")
        custom_model_name = arguments.get("model_name")
        
        # Get experiment details
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM model_experiments WHERE experiment_id = ?", (experiment_id,))
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return [mcp_types.TextContent(
                type="text",
                text=f"Error: Experiment {experiment_id} not found"
            )]
        
        # Check if model artifacts exist
        model_dir = row["model_artifact_path"]
        if not model_dir or not os.path.exists(model_dir):
            return [mcp_types.TextContent(
                type="text",
                text=f"Error: Model artifacts not found for experiment {experiment_id}"
            )]
        
        # Load existing model artifacts
        artifact_result = load_model_artifacts(model_dir)
        if not artifact_result["success"]:
            return [mcp_types.TextContent(
                type="text",
                text=f"Error loading model artifacts: {artifact_result['message']}"
            )]
        
        model = artifact_result["model"]
        scaler = artifact_result["scaler"]
        
        # Determine model name
        model_name = custom_model_name or f"{row['model_type']}_demand_forecast"
        
        # Generate version
        if version_arg == "auto":
            existing_versions = get_existing_model_versions(model_name)
            version = generate_model_version(row["model_type"], existing_versions)
        else:
            version = version_arg
        
        # Prepare performance metrics
        performance_metrics = {
            "accuracy_score": row["accuracy_score"],
            "mape_score": row["mape_score"],
            "rmse_score": row["rmse_score"]
        }
        
        hyperparameters = json.loads(row["hyperparameters"]) if row["hyperparameters"] else {}
        
        # Create comprehensive model artifacts package
        artifacts = create_model_artifacts_package(
            model=model,
            scaler=scaler,
            model_type=row["model_type"],
            hyperparameters=hyperparameters,
            performance_metrics=performance_metrics,
            version=version,
            model_name=model_name
        )
        
        # Create model registry entry
        model_registry = ModelRegistry(
            model_id=artifacts.model_id,
            model_name=model_name,
            version=version,
            model_type=row["model_type"],
            status=ModelStatus.VALIDATION,
            created_at=datetime.now(),
            performance_metrics=performance_metrics,
            artifact_location=artifacts.artifact_location,
            hyperparameters=hyperparameters,
            training_data_hash=row["training_data_hash"]
        )
        
        # Register model
        register_result = register_model(model_registry)
        
        if register_result["success"]:
            # Track performance history
            track_result = track_model_performance_history(artifacts.model_id, performance_metrics)
            
            result = {
                "success": True,
                "model_id": artifacts.model_id,
                "model_name": model_name,
                "version": version,
                "artifact_location": artifacts.artifact_location,
                "performance_metrics": performance_metrics,
                "registry_id": register_result["registry_id"],
                "status": "VALIDATION",
                "artifact_structure": artifacts.metadata["artifact_structure"],
                "performance_tracking": track_result,
                "deployment_ready": True
            }
        else:
            result = {
                "success": False,
                "message": f"Failed to register model: {register_result['message']}"
            }
        
        return [mcp_types.TextContent(
            type="text",
            text=json.dumps(result, indent=2)
        )]
        
    except Exception as e:
        return [mcp_types.TextContent(
            type="text",
            text=f"Error generating model artifacts: {e}"
        )]


async def main():
    """Main function to run the MCP server."""
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="forecasting-model-mcp",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


if __name__ == "__main__":
    asyncio.run(main())