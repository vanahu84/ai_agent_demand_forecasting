"""
Core data models for autonomous demand forecasting system.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import json


class SeverityLevel(Enum):
    """Drift severity classification levels."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class ModelStatus(Enum):
    """Model lifecycle status."""
    TRAINING = "TRAINING"
    VALIDATION = "VALIDATION"
    PRODUCTION = "PRODUCTION"
    RETIRED = "RETIRED"


class ValidationStatus(Enum):
    """Model validation status."""
    PASSED = "PASSED"
    FAILED = "FAILED"
    PENDING = "PENDING"


class DeploymentStatus(Enum):
    """Deployment status."""
    PENDING = "PENDING"
    DEPLOYING = "DEPLOYING"
    ACTIVE = "ACTIVE"
    ROLLED_BACK = "ROLLED_BACK"
    FAILED = "FAILED"


class WorkflowStatus(Enum):
    """Retraining workflow status."""
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


@dataclass
class Product:
    """Product data model."""
    id: str
    name: str
    category: str
    subcategory: Optional[str] = None
    brand: Optional[str] = None
    unit_price: Optional[float] = None
    cost: Optional[float] = None
    active: bool = True
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


@dataclass
class Store:
    """Store data model."""
    id: str
    name: str
    location: str
    region: Optional[str] = None
    store_type: Optional[str] = None
    square_footage: Optional[int] = None
    active: bool = True
    created_at: Optional[datetime] = None


@dataclass
class SalesTransaction:
    """Sales transaction data model."""
    transaction_id: str
    product_id: str
    store_id: str
    category: str
    quantity: int
    unit_price: float
    total_amount: float
    transaction_date: datetime
    customer_segment: Optional[str] = None
    promotion_applied: bool = False
    id: Optional[int] = None
    created_at: Optional[datetime] = None


@dataclass
class CustomerBehavior:
    """Customer behavior analysis data model."""
    customer_segment: str
    product_category: str
    analysis_date: datetime
    avg_purchase_frequency: Optional[float] = None
    seasonal_multiplier: Optional[float] = None
    price_sensitivity: Optional[float] = None
    id: Optional[int] = None
    created_at: Optional[datetime] = None


@dataclass
class InventoryLevel:
    """Inventory level data model."""
    product_id: str
    store_id: str
    current_stock: int
    available_stock: int
    last_updated: datetime
    reserved_stock: int = 0
    reorder_point: Optional[int] = None
    max_stock_level: Optional[int] = None
    id: Optional[int] = None


@dataclass
class StockoutEvent:
    """Stockout event data model."""
    product_id: str
    store_id: str
    stockout_date: datetime
    duration_hours: Optional[int] = None
    lost_sales_estimate: Optional[float] = None
    restock_date: Optional[datetime] = None
    id: Optional[int] = None
    created_at: Optional[datetime] = None


@dataclass
class AccuracyMetrics:
    """Model accuracy metrics data model."""
    model_id: str
    timestamp: datetime
    accuracy_score: Optional[float] = None
    mape_score: Optional[float] = None
    rmse_score: Optional[float] = None
    prediction_count: Optional[int] = None
    product_categories: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'model_id': self.model_id,
            'timestamp': self.timestamp.isoformat(),
            'accuracy_score': self.accuracy_score,
            'mape_score': self.mape_score,
            'rmse_score': self.rmse_score,
            'prediction_count': self.prediction_count,
            'product_categories': self.product_categories
        }


@dataclass
class DriftEvent:
    """Model drift event data model."""
    model_id: str
    severity: SeverityLevel
    detected_at: datetime
    accuracy_drop: float
    affected_categories: List[str] = field(default_factory=list)
    drift_score: Optional[float] = None
    resolved_at: Optional[datetime] = None
    drift_analysis: Optional[Dict[str, Any]] = None
    id: Optional[int] = None
    created_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'model_id': self.model_id,
            'severity': self.severity.value,
            'detected_at': self.detected_at.isoformat(),
            'accuracy_drop': self.accuracy_drop,
            'affected_categories': self.affected_categories,
            'drift_score': self.drift_score,
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None,
            'drift_analysis': self.drift_analysis
        }


@dataclass
class ModelExperiment:
    """Model experiment data model."""
    experiment_id: str
    model_type: str
    training_start: datetime
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    training_data_hash: Optional[str] = None
    training_end: Optional[datetime] = None
    accuracy_score: Optional[float] = None
    mape_score: Optional[float] = None
    rmse_score: Optional[float] = None
    model_artifact_path: Optional[str] = None
    experiment_notes: Optional[str] = None
    id: Optional[int] = None
    created_at: Optional[datetime] = None
    
    @property
    def training_duration(self) -> Optional[timedelta]:
        """Calculate training duration."""
        if self.training_end and self.training_start:
            return self.training_end - self.training_start
        return None


@dataclass
class ModelRegistry:
    """Model registry entry data model."""
    model_id: str
    model_name: str
    version: str
    model_type: str
    created_at: datetime
    status: ModelStatus = ModelStatus.TRAINING
    deployed_at: Optional[datetime] = None
    retired_at: Optional[datetime] = None
    performance_metrics: Optional[Dict[str, Any]] = None
    artifact_location: Optional[str] = None
    hyperparameters: Optional[Dict[str, Any]] = None
    training_data_hash: Optional[str] = None
    id: Optional[int] = None


@dataclass
class ValidationResult:
    """Model validation result data model."""
    model_id: str
    validation_dataset_id: str
    validation_date: datetime
    validation_status: ValidationStatus = ValidationStatus.PENDING
    accuracy_score: Optional[float] = None
    baseline_accuracy: Optional[float] = None
    improvement_percentage: Optional[float] = None
    statistical_significance: Optional[float] = None
    validation_notes: Optional[str] = None
    id: Optional[int] = None
    created_at: Optional[datetime] = None
    
    @property
    def passed(self) -> bool:
        """Check if validation passed."""
        return self.validation_status == ValidationStatus.PASSED


@dataclass
class HoldoutDataset:
    """Holdout dataset data model."""
    dataset_id: str
    created_at: datetime
    product_categories: List[str] = field(default_factory=list)
    date_range_start: Optional[datetime] = None
    date_range_end: Optional[datetime] = None
    sample_size: Optional[int] = None
    dataset_path: Optional[str] = None
    id: Optional[int] = None


@dataclass
class DeploymentResult:
    """Deployment result data model."""
    deployment_id: str
    model_id: str
    started_at: datetime
    status: DeploymentStatus = DeploymentStatus.PENDING
    deployment_strategy: str = "blue_green"
    completed_at: Optional[datetime] = None
    rollback_at: Optional[datetime] = None
    performance_metrics: Optional[Dict[str, float]] = None
    deployment_notes: Optional[str] = None
    id: Optional[int] = None
    created_at: Optional[datetime] = None


@dataclass
class ProductionMonitoring:
    """Production monitoring data model."""
    deployment_id: str
    timestamp: datetime
    accuracy_score: Optional[float] = None
    prediction_latency: Optional[float] = None
    error_rate: Optional[float] = None
    throughput: Optional[int] = None
    alert_triggered: bool = False
    id: Optional[int] = None
    created_at: Optional[datetime] = None


@dataclass
class RetrainingWorkflow:
    """Retraining workflow data model."""
    workflow_id: str
    trigger_reason: str
    started_at: datetime
    status: WorkflowStatus = WorkflowStatus.RUNNING
    completed_at: Optional[datetime] = None
    models_trained: int = 0
    models_deployed: int = 0
    business_impact_score: Optional[float] = None
    workflow_metadata: Optional[Dict[str, Any]] = None
    id: Optional[int] = None
    created_at: Optional[datetime] = None


@dataclass
class BusinessImpact:
    """Business impact analysis data model."""
    deployment_id: str
    metric_type: str
    calculated_at: datetime
    baseline_value: Optional[float] = None
    improved_value: Optional[float] = None
    improvement_percentage: Optional[float] = None
    revenue_impact: Optional[float] = None
    id: Optional[int] = None
    created_at: Optional[datetime] = None


# Composite data structures for complex operations

@dataclass
class TrainingData:
    """Training dataset collection."""
    dataset_id: str
    date_range: Tuple[datetime, datetime]
    sales_data: List[SalesTransaction] = field(default_factory=list)
    inventory_data: List[InventoryLevel] = field(default_factory=list)
    customer_behavior: List[CustomerBehavior] = field(default_factory=list)
    quality_score: Optional[float] = None
    
    @property
    def total_transactions(self) -> int:
        """Get total number of sales transactions."""
        return len(self.sales_data)
    
    @property
    def product_categories(self) -> List[str]:
        """Get unique product categories in the dataset."""
        return list(set(tx.category for tx in self.sales_data))


@dataclass
class DriftAnalysis:
    """Comprehensive drift analysis result."""
    model_id: str
    analysis_timestamp: datetime
    drift_events: List[DriftEvent] = field(default_factory=list)
    accuracy_trends: Dict[str, List[float]] = field(default_factory=dict)
    affected_categories: List[str] = field(default_factory=list)
    recommended_actions: List[str] = field(default_factory=list)
    
    @property
    def highest_severity(self) -> Optional[SeverityLevel]:
        """Get the highest severity level among drift events."""
        if not self.drift_events:
            return None
        severities = [event.severity for event in self.drift_events]
        if SeverityLevel.HIGH in severities:
            return SeverityLevel.HIGH
        elif SeverityLevel.MEDIUM in severities:
            return SeverityLevel.MEDIUM
        else:
            return SeverityLevel.LOW


@dataclass
class ModelArtifacts:
    """Model artifacts collection."""
    model_id: str
    artifact_location: str
    model_type: str
    version: str
    performance_metrics: AccuracyMetrics
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[datetime] = None


@dataclass
class RetrainingPlan:
    """Retraining execution plan."""
    workflow_id: str
    trigger_reason: str
    affected_models: List[str] = field(default_factory=list)
    data_collection_requirements: Dict[str, Any] = field(default_factory=dict)
    training_parameters: Dict[str, Any] = field(default_factory=dict)
    validation_criteria: Dict[str, Any] = field(default_factory=dict)
    deployment_strategy: str = "blue_green"
    estimated_duration: Optional[timedelta] = None


@dataclass
class ImpactReport:
    """Business impact report."""
    report_id: str
    generated_at: datetime
    time_period: Tuple[datetime, datetime]
    model_improvements: List[Dict[str, Any]] = field(default_factory=list)
    business_metrics: List[BusinessImpact] = field(default_factory=list)
    total_revenue_impact: Optional[float] = None
    accuracy_improvements: Dict[str, float] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)