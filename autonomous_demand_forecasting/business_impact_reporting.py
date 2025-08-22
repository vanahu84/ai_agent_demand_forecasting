"""
Business Impact Reporting and Executive Dashboard System
Provides comprehensive business impact analysis, ROI reporting, and executive dashboards.
"""

import json
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from pathlib import Path

from .database.connection import DatabaseConnection
from .database.utils import RetailDatabaseUtils

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ReportPeriod(Enum):
    """Report time periods"""
    DAILY = "DAILY"
    WEEKLY = "WEEKLY"
    MONTHLY = "MONTHLY"
    QUARTERLY = "QUARTERLY"
    YEARLY = "YEARLY"

class ImpactMetric(Enum):
    """Business impact metrics"""
    ACCURACY_IMPROVEMENT = "ACCURACY_IMPROVEMENT"
    REVENUE_IMPACT = "REVENUE_IMPACT"
    COST_SAVINGS = "COST_SAVINGS"
    INVENTORY_OPTIMIZATION = "INVENTORY_OPTIMIZATION"
    FORECAST_RELIABILITY = "FORECAST_RELIABILITY"
    OPERATIONAL_EFFICIENCY = "OPERATIONAL_EFFICIENCY"

@dataclass
class BusinessImpactMetrics:
    """Business impact metrics data model"""
    period_start: datetime
    period_end: datetime
    total_revenue_impact: float
    total_cost_savings: float
    accuracy_improvement_avg: float
    inventory_optimization_value: float
    forecast_reliability_score: float
    operational_efficiency_gain: float
    roi_percentage: float
    models_improved: int
    successful_deployments: int
    total_predictions: int

@dataclass
class ModelPerformanceReport:
    """Model performance report data model"""
    model_id: str
    model_name: str
    deployment_date: datetime
    baseline_accuracy: float
    current_accuracy: float
    accuracy_improvement: float
    revenue_impact: float
    cost_savings: float
    predictions_made: int
    drift_events: int
    retraining_count: int
    uptime_percentage: float

@dataclass
class ExecutiveSummary:
    """Executive summary data model"""
    report_period: str
    total_models: int
    active_models: int
    total_revenue_impact: float
    total_cost_savings: float
    overall_roi: float
    key_achievements: List[str]
    key_challenges: List[str]
    recommendations: List[str]
    trend_analysis: Dict[str, Any]

@dataclass
class AuditLogEntry:
    """Audit log entry data model"""
    timestamp: datetime
    event_type: str
    user_id: Optional[str]
    model_id: Optional[str]
    action: str
    details: Dict[str, Any]
    ip_address: Optional[str]
    session_id: Optional[str]

class BusinessImpactAnalyzer:
    """Analyzes business impact of autonomous demand forecasting system"""
    
    def __init__(self, db_path: str = "autonomous_demand_forecasting.db"):
        self.db_path = db_path
        self.db_utils = RetailDatabaseUtils(db_path)
        
        # Business impact calculation parameters
        self.revenue_per_accuracy_point = 10000.0  # Revenue impact per 1% accuracy improvement
        self.cost_per_prediction = 0.01  # Cost per prediction
        self.inventory_cost_factor = 0.15  # Inventory carrying cost factor
        
        self._initialize_audit_tables()
    
    def _initialize_audit_tables(self):
        """Initialize audit logging tables"""
        db_conn = DatabaseConnection(self.db_path)
        with db_conn.transaction() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY,
                    timestamp DATETIME NOT NULL,
                    event_type TEXT NOT NULL,
                    user_id TEXT,
                    model_id TEXT,
                    action TEXT NOT NULL,
                    details TEXT,
                    ip_address TEXT,
                    session_id TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS business_metrics_cache (
                    id INTEGER PRIMARY KEY,
                    metric_type TEXT NOT NULL,
                    period_start DATETIME NOT NULL,
                    period_end DATETIME NOT NULL,
                    metric_value REAL NOT NULL,
                    metadata TEXT,
                    calculated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes for performance
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_audit_log_timestamp 
                ON audit_log(timestamp)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_audit_log_model_id 
                ON audit_log(model_id)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_business_metrics_period 
                ON business_metrics_cache(period_start, period_end)
            """)
    
    def calculate_business_impact_metrics(self, period: ReportPeriod, 
                                        start_date: Optional[datetime] = None) -> BusinessImpactMetrics:
        """Calculate comprehensive business impact metrics for a period"""
        
        # Determine period dates
        if start_date is None:
            start_date = datetime.now()
        
        period_start, period_end = self._get_period_dates(period, start_date)
        
        logger.info(f"Calculating business impact metrics for {period.value} period: {period_start} to {period_end}")
        
        # Get database connection
        db_conn = DatabaseConnection(self.db_path)
        
        # Calculate revenue impact
        total_revenue_impact = self._calculate_revenue_impact(db_conn, period_start, period_end)
        
        # Calculate cost savings
        total_cost_savings = self._calculate_cost_savings(db_conn, period_start, period_end)
        
        # Calculate accuracy improvements
        accuracy_improvement_avg = self._calculate_accuracy_improvement(db_conn, period_start, period_end)
        
        # Calculate inventory optimization value
        inventory_optimization_value = self._calculate_inventory_optimization(db_conn, period_start, period_end)
        
        # Calculate forecast reliability
        forecast_reliability_score = self._calculate_forecast_reliability(db_conn, period_start, period_end)
        
        # Calculate operational efficiency
        operational_efficiency_gain = self._calculate_operational_efficiency(db_conn, period_start, period_end)
        
        # Calculate ROI
        total_investment = self._calculate_total_investment(db_conn, period_start, period_end)
        total_benefit = total_revenue_impact + total_cost_savings
        roi_percentage = (total_benefit / total_investment * 100) if total_investment > 0 else 0
        
        # Get operational metrics
        models_improved = self._count_models_improved(db_conn, period_start, period_end)
        successful_deployments = self._count_successful_deployments(db_conn, period_start, period_end)
        total_predictions = self._count_total_predictions(db_conn, period_start, period_end)
        
        return BusinessImpactMetrics(
            period_start=period_start,
            period_end=period_end,
            total_revenue_impact=total_revenue_impact,
            total_cost_savings=total_cost_savings,
            accuracy_improvement_avg=accuracy_improvement_avg,
            inventory_optimization_value=inventory_optimization_value,
            forecast_reliability_score=forecast_reliability_score,
            operational_efficiency_gain=operational_efficiency_gain,
            roi_percentage=roi_percentage,
            models_improved=models_improved,
            successful_deployments=successful_deployments,
            total_predictions=total_predictions
        )
    
    def generate_model_performance_report(self, model_id: Optional[str] = None) -> List[ModelPerformanceReport]:
        """Generate detailed model performance reports"""
        
        db_conn = DatabaseConnection(self.db_path)
        
        # Build query for model(s)
        if model_id:
            model_filter = "WHERE mr.model_id = ?"
            params = (model_id,)
        else:
            model_filter = "WHERE mr.status = 'PRODUCTION'"
            params = ()
        
        query = f"""
        SELECT 
            mr.model_id,
            mr.model_name,
            mr.deployed_at,
            mr.performance_metrics
        FROM model_registry mr
        {model_filter}
        ORDER BY mr.deployed_at DESC
        """
        
        results = db_conn.execute_query(query, params)
        
        reports = []
        for row in results:
            model_id = row['model_id']
            model_name = row['model_name']
            deployed_at = datetime.fromisoformat(row['deployed_at']) if row['deployed_at'] else datetime.now()
            
            # Get performance metrics
            baseline_accuracy, current_accuracy = self._get_model_accuracy_metrics(db_conn, model_id)
            accuracy_improvement = current_accuracy - baseline_accuracy
            
            # Get business impact
            revenue_impact = self._get_model_revenue_impact(db_conn, model_id)
            cost_savings = self._get_model_cost_savings(db_conn, model_id)
            
            # Get operational metrics
            predictions_made = self._get_model_prediction_count(db_conn, model_id)
            drift_events = self._get_model_drift_events(db_conn, model_id)
            retraining_count = self._get_model_retraining_count(db_conn, model_id)
            uptime_percentage = self._get_model_uptime(db_conn, model_id)
            
            reports.append(ModelPerformanceReport(
                model_id=model_id,
                model_name=model_name,
                deployment_date=deployed_at,
                baseline_accuracy=baseline_accuracy,
                current_accuracy=current_accuracy,
                accuracy_improvement=accuracy_improvement,
                revenue_impact=revenue_impact,
                cost_savings=cost_savings,
                predictions_made=predictions_made,
                drift_events=drift_events,
                retraining_count=retraining_count,
                uptime_percentage=uptime_percentage
            ))
        
        return reports
    
    def generate_executive_summary(self, period: ReportPeriod) -> ExecutiveSummary:
        """Generate executive summary report"""
        
        # Get business impact metrics
        impact_metrics = self.calculate_business_impact_metrics(period)
        
        # Get model performance reports
        model_reports = self.generate_model_performance_report()
        
        # Calculate summary statistics
        total_models = len(model_reports)
        active_models = len([r for r in model_reports if r.uptime_percentage > 95])
        
        # Generate key achievements
        key_achievements = self._generate_key_achievements(impact_metrics, model_reports)
        
        # Generate key challenges
        key_challenges = self._generate_key_challenges(model_reports)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(impact_metrics, model_reports)
        
        # Generate trend analysis
        trend_analysis = self._generate_trend_analysis(period)
        
        return ExecutiveSummary(
            report_period=f"{period.value} ({impact_metrics.period_start.strftime('%Y-%m-%d')} to {impact_metrics.period_end.strftime('%Y-%m-%d')})",
            total_models=total_models,
            active_models=active_models,
            total_revenue_impact=impact_metrics.total_revenue_impact,
            total_cost_savings=impact_metrics.total_cost_savings,
            overall_roi=impact_metrics.roi_percentage,
            key_achievements=key_achievements,
            key_challenges=key_challenges,
            recommendations=recommendations,
            trend_analysis=trend_analysis
        )
    
    def log_audit_event(self, event_type: str, action: str, user_id: Optional[str] = None,
                       model_id: Optional[str] = None, details: Optional[Dict[str, Any]] = None,
                       ip_address: Optional[str] = None, session_id: Optional[str] = None):
        """Log audit event for compliance and tracking"""
        
        audit_entry = AuditLogEntry(
            timestamp=datetime.now(),
            event_type=event_type,
            user_id=user_id,
            model_id=model_id,
            action=action,
            details=details or {},
            ip_address=ip_address,
            session_id=session_id
        )
        
        db_conn = DatabaseConnection(self.db_path)
        with db_conn.transaction() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO audit_log 
                (timestamp, event_type, user_id, model_id, action, details, ip_address, session_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                audit_entry.timestamp,
                audit_entry.event_type,
                audit_entry.user_id,
                audit_entry.model_id,
                audit_entry.action,
                json.dumps(audit_entry.details),
                audit_entry.ip_address,
                audit_entry.session_id
            ))
        
        logger.info(f"Audit event logged: {event_type} - {action}")
    
    def get_audit_log(self, start_date: Optional[datetime] = None, 
                     end_date: Optional[datetime] = None,
                     event_type: Optional[str] = None,
                     model_id: Optional[str] = None) -> List[AuditLogEntry]:
        """Retrieve audit log entries with optional filtering"""
        
        db_conn = DatabaseConnection(self.db_path)
        
        # Build query with filters
        query = "SELECT * FROM audit_log WHERE 1=1"
        params = []
        
        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date)
        
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date)
        
        if event_type:
            query += " AND event_type = ?"
            params.append(event_type)
        
        if model_id:
            query += " AND model_id = ?"
            params.append(model_id)
        
        query += " ORDER BY timestamp DESC"
        
        results = db_conn.execute_query(query, tuple(params))
        
        audit_entries = []
        for row in results:
            audit_entries.append(AuditLogEntry(
                timestamp=datetime.fromisoformat(row['timestamp']),
                event_type=row['event_type'],
                user_id=row['user_id'],
                model_id=row['model_id'],
                action=row['action'],
                details=json.loads(row['details']) if row['details'] else {},
                ip_address=row['ip_address'],
                session_id=row['session_id']
            ))
        
        return audit_entries
    
    def _get_period_dates(self, period: ReportPeriod, reference_date: datetime) -> Tuple[datetime, datetime]:
        """Get start and end dates for a reporting period"""
        
        if period == ReportPeriod.DAILY:
            start = reference_date.replace(hour=0, minute=0, second=0, microsecond=0)
            end = start + timedelta(days=1) - timedelta(microseconds=1)
        elif period == ReportPeriod.WEEKLY:
            days_since_monday = reference_date.weekday()
            start = (reference_date - timedelta(days=days_since_monday)).replace(hour=0, minute=0, second=0, microsecond=0)
            end = start + timedelta(days=7) - timedelta(microseconds=1)
        elif period == ReportPeriod.MONTHLY:
            start = reference_date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            if start.month == 12:
                end = start.replace(year=start.year + 1, month=1) - timedelta(microseconds=1)
            else:
                end = start.replace(month=start.month + 1) - timedelta(microseconds=1)
        elif period == ReportPeriod.QUARTERLY:
            quarter_start_month = ((reference_date.month - 1) // 3) * 3 + 1
            start = reference_date.replace(month=quarter_start_month, day=1, hour=0, minute=0, second=0, microsecond=0)
            if quarter_start_month == 10:
                end = start.replace(year=start.year + 1, month=1) - timedelta(microseconds=1)
            else:
                end = start.replace(month=quarter_start_month + 3) - timedelta(microseconds=1)
        elif period == ReportPeriod.YEARLY:
            start = reference_date.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
            end = start.replace(year=start.year + 1) - timedelta(microseconds=1)
        else:
            raise ValueError(f"Unsupported period: {period}")
        
        return start, end
    
    def _calculate_revenue_impact(self, db_conn: DatabaseConnection, start_date: datetime, end_date: datetime) -> float:
        """Calculate total revenue impact from model improvements"""
        
        query = """
        SELECT SUM(revenue_impact) as total_revenue
        FROM business_impact 
        WHERE calculated_at BETWEEN ? AND ?
        """
        
        result = db_conn.execute_query(query, (start_date, end_date))
        return result[0]['total_revenue'] if result and result[0]['total_revenue'] else 0.0
    
    def _calculate_cost_savings(self, db_conn: DatabaseConnection, start_date: datetime, end_date: datetime) -> float:
        """Calculate total cost savings from operational efficiency"""
        
        # Calculate cost savings from reduced manual interventions
        query = """
        SELECT COUNT(*) as automated_actions
        FROM retraining_workflows 
        WHERE started_at BETWEEN ? AND ? AND status = 'COMPLETED'
        """
        
        result = db_conn.execute_query(query, (start_date, end_date))
        automated_actions = result[0]['automated_actions'] if result else 0
        
        # Estimate cost savings (assuming $500 per manual intervention avoided)
        cost_per_manual_action = 500.0
        return automated_actions * cost_per_manual_action
    
    def _calculate_accuracy_improvement(self, db_conn: DatabaseConnection, start_date: datetime, end_date: datetime) -> float:
        """Calculate average accuracy improvement across models"""
        
        query = """
        SELECT AVG(improvement_percentage) as avg_improvement
        FROM business_impact 
        WHERE calculated_at BETWEEN ? AND ? AND metric_type = 'accuracy_improvement'
        """
        
        result = db_conn.execute_query(query, (start_date, end_date))
        return result[0]['avg_improvement'] if result and result[0]['avg_improvement'] else 0.0
    
    def _calculate_inventory_optimization(self, db_conn: DatabaseConnection, start_date: datetime, end_date: datetime) -> float:
        """Calculate inventory optimization value"""
        
        # Simplified calculation based on stockout reduction
        query = """
        SELECT COUNT(*) as stockout_events
        FROM stockout_events 
        WHERE stockout_date BETWEEN ? AND ?
        """
        
        result = db_conn.execute_query(query, (start_date, end_date))
        stockout_events = result[0]['stockout_events'] if result else 0
        
        # Estimate value (assuming $1000 per stockout event avoided)
        value_per_stockout_avoided = 1000.0
        baseline_stockouts = 50  # Baseline monthly stockouts before system
        
        # Calculate improvement (simplified)
        stockouts_avoided = max(0, baseline_stockouts - stockout_events)
        return stockouts_avoided * value_per_stockout_avoided
    
    def _calculate_forecast_reliability(self, db_conn: DatabaseConnection, start_date: datetime, end_date: datetime) -> float:
        """Calculate forecast reliability score"""
        
        query = """
        SELECT AVG(accuracy_score) as avg_accuracy
        FROM model_performance 
        WHERE timestamp BETWEEN ? AND ?
        """
        
        result = db_conn.execute_query(query, (start_date, end_date))
        return result[0]['avg_accuracy'] if result and result[0]['avg_accuracy'] else 0.0
    
    def _calculate_operational_efficiency(self, db_conn: DatabaseConnection, start_date: datetime, end_date: datetime) -> float:
        """Calculate operational efficiency gain"""
        
        # Calculate based on successful automated workflows
        query = """
        SELECT 
            COUNT(*) as total_workflows,
            SUM(CASE WHEN status = 'COMPLETED' THEN 1 ELSE 0 END) as successful_workflows
        FROM retraining_workflows 
        WHERE started_at BETWEEN ? AND ?
        """
        
        result = db_conn.execute_query(query, (start_date, end_date))
        if result and result[0]['total_workflows'] > 0:
            success_rate = result[0]['successful_workflows'] / result[0]['total_workflows']
            return success_rate * 100  # Return as percentage
        
        return 0.0
    
    def _calculate_total_investment(self, db_conn: DatabaseConnection, start_date: datetime, end_date: datetime) -> float:
        """Calculate total investment in the system"""
        
        # Simplified calculation based on compute costs and operational overhead
        query = """
        SELECT COUNT(*) as total_predictions
        FROM model_performance 
        WHERE timestamp BETWEEN ? AND ?
        """
        
        result = db_conn.execute_query(query, (start_date, end_date))
        total_predictions = result[0]['total_predictions'] if result else 0
        
        # Calculate investment (compute costs + operational overhead)
        compute_cost = total_predictions * self.cost_per_prediction
        operational_overhead = 5000.0  # Monthly operational overhead
        
        return compute_cost + operational_overhead
    
    def _count_models_improved(self, db_conn: DatabaseConnection, start_date: datetime, end_date: datetime) -> int:
        """Count number of models that were improved in the period"""
        
        query = """
        SELECT COUNT(DISTINCT model_id) as improved_models
        FROM business_impact 
        WHERE calculated_at BETWEEN ? AND ? AND improvement_percentage > 0
        """
        
        result = db_conn.execute_query(query, (start_date, end_date))
        return result[0]['improved_models'] if result else 0
    
    def _count_successful_deployments(self, db_conn: DatabaseConnection, start_date: datetime, end_date: datetime) -> int:
        """Count successful deployments in the period"""
        
        query = """
        SELECT COUNT(*) as successful_deployments
        FROM deployments 
        WHERE started_at BETWEEN ? AND ? AND status = 'ACTIVE'
        """
        
        result = db_conn.execute_query(query, (start_date, end_date))
        return result[0]['successful_deployments'] if result else 0
    
    def _count_total_predictions(self, db_conn: DatabaseConnection, start_date: datetime, end_date: datetime) -> int:
        """Count total predictions made in the period"""
        
        query = """
        SELECT SUM(prediction_count) as total_predictions
        FROM model_performance 
        WHERE timestamp BETWEEN ? AND ?
        """
        
        result = db_conn.execute_query(query, (start_date, end_date))
        return result[0]['total_predictions'] if result and result[0]['total_predictions'] else 0
    
    def _get_model_accuracy_metrics(self, db_conn: DatabaseConnection, model_id: str) -> Tuple[float, float]:
        """Get baseline and current accuracy for a model"""
        
        # Get baseline accuracy (first recorded accuracy)
        baseline_query = """
        SELECT accuracy_score 
        FROM model_performance 
        WHERE model_id = ? 
        ORDER BY timestamp ASC 
        LIMIT 1
        """
        
        baseline_result = db_conn.execute_query(baseline_query, (model_id,))
        baseline_accuracy = baseline_result[0]['accuracy_score'] if baseline_result else 0.0
        
        # Get current accuracy (latest recorded accuracy)
        current_query = """
        SELECT accuracy_score 
        FROM model_performance 
        WHERE model_id = ? 
        ORDER BY timestamp DESC 
        LIMIT 1
        """
        
        current_result = db_conn.execute_query(current_query, (model_id,))
        current_accuracy = current_result[0]['accuracy_score'] if current_result else 0.0
        
        return baseline_accuracy, current_accuracy
    
    def _get_model_revenue_impact(self, db_conn: DatabaseConnection, model_id: str) -> float:
        """Get total revenue impact for a model"""
        
        query = """
        SELECT SUM(revenue_impact) as total_revenue
        FROM business_impact 
        WHERE deployment_id IN (
            SELECT deployment_id FROM deployments WHERE model_id = ?
        )
        """
        
        result = db_conn.execute_query(query, (model_id,))
        return result[0]['total_revenue'] if result and result[0]['total_revenue'] else 0.0
    
    def _get_model_cost_savings(self, db_conn: DatabaseConnection, model_id: str) -> float:
        """Get cost savings for a model"""
        
        # Simplified calculation based on retraining efficiency
        query = """
        SELECT COUNT(*) as retraining_count
        FROM retraining_workflows 
        WHERE workflow_id LIKE ? AND status = 'COMPLETED'
        """
        
        result = db_conn.execute_query(query, (f"%{model_id}%",))
        retraining_count = result[0]['retraining_count'] if result else 0
        
        # Estimate cost savings per successful retraining
        cost_savings_per_retraining = 200.0
        return retraining_count * cost_savings_per_retraining
    
    def _get_model_prediction_count(self, db_conn: DatabaseConnection, model_id: str) -> int:
        """Get total prediction count for a model"""
        
        query = """
        SELECT SUM(prediction_count) as total_predictions
        FROM model_performance 
        WHERE model_id = ?
        """
        
        result = db_conn.execute_query(query, (model_id,))
        return result[0]['total_predictions'] if result and result[0]['total_predictions'] else 0
    
    def _get_model_drift_events(self, db_conn: DatabaseConnection, model_id: str) -> int:
        """Get drift event count for a model"""
        
        query = """
        SELECT COUNT(*) as drift_events
        FROM drift_events 
        WHERE model_id = ?
        """
        
        result = db_conn.execute_query(query, (model_id,))
        return result[0]['drift_events'] if result else 0
    
    def _get_model_retraining_count(self, db_conn: DatabaseConnection, model_id: str) -> int:
        """Get retraining count for a model"""
        
        query = """
        SELECT COUNT(*) as retraining_count
        FROM retraining_workflows 
        WHERE workflow_id LIKE ?
        """
        
        result = db_conn.execute_query(query, (f"%{model_id}%",))
        return result[0]['retraining_count'] if result else 0
    
    def _get_model_uptime(self, db_conn: DatabaseConnection, model_id: str) -> float:
        """Calculate model uptime percentage"""
        
        # Simplified calculation based on deployment status
        query = """
        SELECT status, started_at, completed_at
        FROM deployments 
        WHERE model_id = ? 
        ORDER BY started_at DESC 
        LIMIT 1
        """
        
        result = db_conn.execute_query(query, (model_id,))
        if result and result[0]['status'] == 'ACTIVE':
            return 99.5  # Assume high uptime for active deployments
        elif result and result[0]['status'] == 'FAILED':
            return 85.0  # Lower uptime for failed deployments
        else:
            return 95.0  # Default uptime
    
    def _generate_key_achievements(self, impact_metrics: BusinessImpactMetrics, 
                                 model_reports: List[ModelPerformanceReport]) -> List[str]:
        """Generate key achievements for executive summary"""
        
        achievements = []
        
        if impact_metrics.total_revenue_impact > 0:
            achievements.append(f"Generated ${impact_metrics.total_revenue_impact:,.0f} in revenue impact")
        
        if impact_metrics.roi_percentage > 0:
            achievements.append(f"Achieved {impact_metrics.roi_percentage:.1f}% ROI on AI investment")
        
        if impact_metrics.accuracy_improvement_avg > 0:
            achievements.append(f"Improved forecast accuracy by {impact_metrics.accuracy_improvement_avg:.1f}% on average")
        
        if impact_metrics.successful_deployments > 0:
            achievements.append(f"Successfully deployed {impact_metrics.successful_deployments} model updates")
        
        if impact_metrics.operational_efficiency_gain > 90:
            achievements.append(f"Maintained {impact_metrics.operational_efficiency_gain:.1f}% operational efficiency")
        
        # Model-specific achievements
        top_performer = max(model_reports, key=lambda x: x.accuracy_improvement, default=None)
        if top_performer and top_performer.accuracy_improvement > 0.05:
            achievements.append(f"Best performing model ({top_performer.model_name}) improved accuracy by {top_performer.accuracy_improvement:.1%}")
        
        return achievements
    
    def _generate_key_challenges(self, model_reports: List[ModelPerformanceReport]) -> List[str]:
        """Generate key challenges for executive summary"""
        
        challenges = []
        
        # Identify models with high drift events
        high_drift_models = [r for r in model_reports if r.drift_events > 5]
        if high_drift_models:
            challenges.append(f"{len(high_drift_models)} models experienced frequent drift events")
        
        # Identify models with low uptime
        low_uptime_models = [r for r in model_reports if r.uptime_percentage < 95]
        if low_uptime_models:
            challenges.append(f"{len(low_uptime_models)} models had uptime below 95%")
        
        # Identify models with negative accuracy improvement
        declining_models = [r for r in model_reports if r.accuracy_improvement < 0]
        if declining_models:
            challenges.append(f"{len(declining_models)} models showed declining accuracy")
        
        # Identify models requiring frequent retraining
        frequent_retraining = [r for r in model_reports if r.retraining_count > 10]
        if frequent_retraining:
            challenges.append(f"{len(frequent_retraining)} models required frequent retraining")
        
        return challenges
    
    def _generate_recommendations(self, impact_metrics: BusinessImpactMetrics, 
                                model_reports: List[ModelPerformanceReport]) -> List[str]:
        """Generate recommendations for executive summary"""
        
        recommendations = []
        
        if impact_metrics.roi_percentage < 50:
            recommendations.append("Consider optimizing model training costs to improve ROI")
        
        if impact_metrics.accuracy_improvement_avg < 0.05:
            recommendations.append("Investigate data quality issues that may be limiting accuracy improvements")
        
        # Model-specific recommendations
        high_drift_models = [r for r in model_reports if r.drift_events > 5]
        if high_drift_models:
            recommendations.append("Implement more frequent retraining for models with high drift rates")
        
        low_performing_models = [r for r in model_reports if r.accuracy_improvement < 0.02]
        if len(low_performing_models) > len(model_reports) * 0.3:
            recommendations.append("Review feature engineering and model architecture for underperforming models")
        
        if impact_metrics.operational_efficiency_gain < 85:
            recommendations.append("Investigate workflow automation failures to improve operational efficiency")
        
        return recommendations
    
    def _generate_trend_analysis(self, period: ReportPeriod) -> Dict[str, Any]:
        """Generate trend analysis for executive summary"""
        
        # Get historical data for comparison
        current_metrics = self.calculate_business_impact_metrics(period)
        
        # Calculate previous period metrics for comparison
        if period == ReportPeriod.MONTHLY:
            previous_start = current_metrics.period_start - timedelta(days=30)
        elif period == ReportPeriod.QUARTERLY:
            previous_start = current_metrics.period_start - timedelta(days=90)
        elif period == ReportPeriod.YEARLY:
            previous_start = current_metrics.period_start - timedelta(days=365)
        else:
            previous_start = current_metrics.period_start - timedelta(days=7)
        
        previous_metrics = self.calculate_business_impact_metrics(period, previous_start)
        
        # Calculate trends
        revenue_trend = ((current_metrics.total_revenue_impact - previous_metrics.total_revenue_impact) / 
                        max(previous_metrics.total_revenue_impact, 1)) * 100
        
        accuracy_trend = current_metrics.accuracy_improvement_avg - previous_metrics.accuracy_improvement_avg
        
        roi_trend = current_metrics.roi_percentage - previous_metrics.roi_percentage
        
        return {
            "revenue_trend_percentage": revenue_trend,
            "accuracy_trend_points": accuracy_trend,
            "roi_trend_points": roi_trend,
            "models_improved_trend": current_metrics.models_improved - previous_metrics.models_improved,
            "predictions_trend": current_metrics.total_predictions - previous_metrics.total_predictions
        }

# Global business impact analyzer instance
_business_analyzer: Optional[BusinessImpactAnalyzer] = None

def get_business_impact_analyzer(db_path: str = "autonomous_demand_forecasting.db") -> BusinessImpactAnalyzer:
    """Get or create global business impact analyzer instance"""
    global _business_analyzer
    if _business_analyzer is None:
        _business_analyzer = BusinessImpactAnalyzer(db_path)
    return _business_analyzer