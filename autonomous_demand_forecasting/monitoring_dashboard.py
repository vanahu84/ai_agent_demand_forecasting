"""
Real-time Model Performance Dashboard
Provides web-based visualization for model accuracy, drift detection, and retraining status.
"""

import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from flask import Flask, render_template, jsonify, request, redirect, url_for
from flask_socketio import SocketIO, emit
import plotly.graph_objs as go
import plotly.utils
import pandas as pd
import threading
import time
import logging

from .database.connection import DatabaseConnection
from .database.utils import RetailDatabaseUtils

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DashboardMetrics:
    """Real-time dashboard metrics"""
    timestamp: datetime
    active_models: int
    avg_accuracy: float
    drift_alerts: int
    retraining_jobs: int
    deployment_status: str
    business_impact: float

@dataclass
class ModelPerformanceData:
    """Model performance visualization data"""
    model_id: str
    model_name: str
    accuracy_trend: List[float]
    timestamps: List[str]
    current_accuracy: float
    drift_status: str
    last_updated: datetime

@dataclass
class AlertData:
    """Alert information for dashboard"""
    alert_id: str
    severity: str
    message: str
    model_id: str
    timestamp: datetime
    acknowledged: bool

class MonitoringDashboard:
    """Real-time monitoring dashboard for autonomous demand forecasting"""
    
    def __init__(self, db_path: str = "autonomous_demand_forecasting.db"):
        self.db_path = db_path
        self.db_utils = RetailDatabaseUtils(db_path)
        self.app = Flask(__name__, template_folder='templates', static_folder='static')
        self.app.config['SECRET_KEY'] = 'autonomous_forecasting_dashboard_2024'
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        # Dashboard state
        self.active_alerts: List[AlertData] = []
        self.dashboard_metrics: Optional[DashboardMetrics] = None
        self.model_performance_cache: Dict[str, ModelPerformanceData] = {}
        
        # Real-time update thread
        self.update_thread = None
        self.running = False
        
        self._setup_routes()
        self._setup_socketio_events()
    
    def _setup_routes(self):
        """Setup Flask routes for dashboard"""
        
        @self.app.route('/')
        def dashboard():
            """Main dashboard page"""
            return render_template('dashboard.html')
        
        @self.app.route('/api/metrics')
        def get_metrics():
            """Get current dashboard metrics"""
            try:
                metrics = self._get_dashboard_metrics()
                return jsonify(asdict(metrics))
            except Exception as e:
                logger.error(f"Error getting metrics: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/api/models')
        def get_models():
            """Get model performance data"""
            try:
                models = self._get_model_performance_data()
                return jsonify([asdict(model) for model in models])
            except Exception as e:
                logger.error(f"Error getting models: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/api/alerts')
        def get_alerts():
            """Get current alerts"""
            try:
                alerts = self._get_current_alerts()
                return jsonify([asdict(alert) for alert in alerts])
            except Exception as e:
                logger.error(f"Error getting alerts: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/api/alerts/<alert_id>/acknowledge', methods=['POST'])
        def acknowledge_alert(alert_id):
            """Acknowledge an alert"""
            try:
                success = self._acknowledge_alert(alert_id)
                return jsonify({"success": success})
            except Exception as e:
                logger.error(f"Error acknowledging alert: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/api/retraining/trigger', methods=['POST'])
        def trigger_retraining():
            """Manual trigger for model retraining"""
            try:
                model_id = request.json.get('model_id')
                reason = request.json.get('reason', 'Manual trigger from dashboard')
                
                workflow_id = self._trigger_manual_retraining(model_id, reason)
                return jsonify({"workflow_id": workflow_id, "success": True})
            except Exception as e:
                logger.error(f"Error triggering retraining: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/api/deployment/<deployment_id>/rollback', methods=['POST'])
        def rollback_deployment(deployment_id):
            """Rollback a deployment"""
            try:
                success = self._rollback_deployment(deployment_id)
                return jsonify({"success": success})
            except Exception as e:
                logger.error(f"Error rolling back deployment: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/api/charts/accuracy')
        def get_accuracy_chart():
            """Get accuracy trend chart data"""
            try:
                hours_back = int(request.args.get('hours', 24))
                chart_data = self._get_accuracy_chart_data(hours_back)
                return jsonify(chart_data)
            except Exception as e:
                logger.error(f"Error getting accuracy chart: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/api/charts/drift')
        def get_drift_chart():
            """Get drift detection chart data"""
            try:
                days_back = int(request.args.get('days', 7))
                chart_data = self._get_drift_chart_data(days_back)
                return jsonify(chart_data)
            except Exception as e:
                logger.error(f"Error getting drift chart: {e}")
                return jsonify({"error": str(e)}), 500
    
    def _setup_socketio_events(self):
        """Setup SocketIO events for real-time updates"""
        
        @self.socketio.on('connect')
        def handle_connect():
            """Handle client connection"""
            logger.info("Client connected to dashboard")
            emit('status', {'msg': 'Connected to monitoring dashboard'})
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            """Handle client disconnection"""
            logger.info("Client disconnected from dashboard")
        
        @self.socketio.on('request_update')
        def handle_update_request():
            """Handle manual update request"""
            try:
                self._broadcast_updates()
            except Exception as e:
                logger.error(f"Error handling update request: {e}")
                emit('error', {'msg': str(e)})
    
    def _get_dashboard_metrics(self) -> DashboardMetrics:
        """Get current dashboard metrics"""
        with DatabaseConnection(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get active models count
            cursor.execute("""
                SELECT COUNT(*) FROM model_registry 
                WHERE status = 'PRODUCTION'
            """)
            active_models = cursor.fetchone()[0]
            
            # Get average accuracy from last 24 hours
            cursor.execute("""
                SELECT AVG(accuracy_score) FROM model_performance 
                WHERE timestamp > datetime('now', '-24 hours')
            """)
            avg_accuracy_result = cursor.fetchone()[0]
            avg_accuracy = avg_accuracy_result if avg_accuracy_result else 0.0
            
            # Get unresolved drift alerts
            cursor.execute("""
                SELECT COUNT(*) FROM drift_events 
                WHERE resolved_at IS NULL
            """)
            drift_alerts = cursor.fetchone()[0]
            
            # Get running retraining jobs
            cursor.execute("""
                SELECT COUNT(*) FROM retraining_workflows 
                WHERE status = 'RUNNING'
            """)
            retraining_jobs = cursor.fetchone()[0]
            
            # Get latest deployment status
            cursor.execute("""
                SELECT status FROM deployments 
                ORDER BY started_at DESC LIMIT 1
            """)
            deployment_result = cursor.fetchone()
            deployment_status = deployment_result[0] if deployment_result else "NONE"
            
            # Get business impact from last 30 days
            cursor.execute("""
                SELECT SUM(revenue_impact) FROM business_impact 
                WHERE calculated_at > datetime('now', '-30 days')
            """)
            business_impact_result = cursor.fetchone()[0]
            business_impact = business_impact_result if business_impact_result else 0.0
            
            return DashboardMetrics(
                timestamp=datetime.now(),
                active_models=active_models,
                avg_accuracy=round(avg_accuracy, 4),
                drift_alerts=drift_alerts,
                retraining_jobs=retraining_jobs,
                deployment_status=deployment_status,
                business_impact=round(business_impact, 2)
            )
    
    def _get_model_performance_data(self) -> List[ModelPerformanceData]:
        """Get model performance data for visualization"""
        with DatabaseConnection(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get active models
            cursor.execute("""
                SELECT model_id, model_name, status FROM model_registry 
                WHERE status IN ('PRODUCTION', 'VALIDATION')
                ORDER BY deployed_at DESC
            """)
            models = cursor.fetchall()
            
            performance_data = []
            for model_id, model_name, status in models:
                # Get performance trend for last 24 hours
                cursor.execute("""
                    SELECT accuracy_score, timestamp FROM model_performance 
                    WHERE model_id = ? AND timestamp > datetime('now', '-24 hours')
                    ORDER BY timestamp ASC
                """, (model_id,))
                performance_records = cursor.fetchall()
                
                if performance_records:
                    accuracy_trend = [record[0] for record in performance_records]
                    timestamps = [record[1] for record in performance_records]
                    current_accuracy = accuracy_trend[-1] if accuracy_trend else 0.0
                    
                    # Check drift status
                    cursor.execute("""
                        SELECT severity FROM drift_events 
                        WHERE model_id = ? AND resolved_at IS NULL
                        ORDER BY detected_at DESC LIMIT 1
                    """, (model_id,))
                    drift_result = cursor.fetchone()
                    drift_status = drift_result[0] if drift_result else "NORMAL"
                    
                    # Get last updated timestamp
                    cursor.execute("""
                        SELECT MAX(timestamp) FROM model_performance 
                        WHERE model_id = ?
                    """, (model_id,))
                    last_updated_result = cursor.fetchone()[0]
                    last_updated = datetime.fromisoformat(last_updated_result) if last_updated_result else datetime.now()
                    
                    performance_data.append(ModelPerformanceData(
                        model_id=model_id,
                        model_name=model_name,
                        accuracy_trend=accuracy_trend,
                        timestamps=timestamps,
                        current_accuracy=round(current_accuracy, 4),
                        drift_status=drift_status,
                        last_updated=last_updated
                    ))
            
            return performance_data
    
    def _get_current_alerts(self) -> List[AlertData]:
        """Get current active alerts"""
        with DatabaseConnection(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get unresolved drift events as alerts
            cursor.execute("""
                SELECT model_id, severity, detected_at, accuracy_drop 
                FROM drift_events 
                WHERE resolved_at IS NULL
                ORDER BY detected_at DESC
            """)
            drift_events = cursor.fetchall()
            
            alerts = []
            for model_id, severity, detected_at, accuracy_drop in drift_events:
                alert_id = f"drift_{model_id}_{detected_at}"
                message = f"Model {model_id} accuracy dropped by {accuracy_drop:.2%}"
                
                alerts.append(AlertData(
                    alert_id=alert_id,
                    severity=severity,
                    message=message,
                    model_id=model_id,
                    timestamp=datetime.fromisoformat(detected_at),
                    acknowledged=False
                ))
            
            # Get failed deployments as alerts
            cursor.execute("""
                SELECT deployment_id, model_id, started_at 
                FROM deployments 
                WHERE status = 'FAILED' AND started_at > datetime('now', '-24 hours')
                ORDER BY started_at DESC
            """)
            failed_deployments = cursor.fetchall()
            
            for deployment_id, model_id, started_at in failed_deployments:
                alert_id = f"deployment_{deployment_id}"
                message = f"Deployment failed for model {model_id}"
                
                alerts.append(AlertData(
                    alert_id=alert_id,
                    severity="HIGH",
                    message=message,
                    model_id=model_id,
                    timestamp=datetime.fromisoformat(started_at),
                    acknowledged=False
                ))
            
            return alerts
    
    def _acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert"""
        try:
            # For drift alerts, mark as resolved
            if alert_id.startswith("drift_"):
                parts = alert_id.split("_")
                model_id = parts[1]
                detected_at = "_".join(parts[2:])
                
                with DatabaseConnection(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        UPDATE drift_events 
                        SET resolved_at = datetime('now')
                        WHERE model_id = ? AND detected_at = ?
                    """, (model_id, detected_at))
                    conn.commit()
                    return cursor.rowcount > 0
            
            # For deployment alerts, update status
            elif alert_id.startswith("deployment_"):
                deployment_id = alert_id.replace("deployment_", "")
                
                with DatabaseConnection(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        UPDATE deployments 
                        SET deployment_notes = COALESCE(deployment_notes, '') || ' [ACKNOWLEDGED]'
                        WHERE deployment_id = ?
                    """, (deployment_id,))
                    conn.commit()
                    return cursor.rowcount > 0
            
            return False
        except Exception as e:
            logger.error(f"Error acknowledging alert {alert_id}: {e}")
            return False
    
    def _trigger_manual_retraining(self, model_id: str, reason: str) -> str:
        """Trigger manual model retraining"""
        workflow_id = f"manual_{model_id}_{int(datetime.now().timestamp())}"
        
        with DatabaseConnection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO retraining_workflows 
                (workflow_id, trigger_reason, started_at, status)
                VALUES (?, ?, datetime('now'), 'RUNNING')
            """, (workflow_id, f"Manual trigger: {reason}"))
            conn.commit()
        
        logger.info(f"Manual retraining triggered for model {model_id}: {workflow_id}")
        return workflow_id
    
    def _rollback_deployment(self, deployment_id: str) -> bool:
        """Rollback a deployment"""
        try:
            with DatabaseConnection(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE deployments 
                    SET status = 'ROLLED_BACK', rollback_at = datetime('now')
                    WHERE deployment_id = ?
                """, (deployment_id,))
                conn.commit()
                
                success = cursor.rowcount > 0
                if success:
                    logger.info(f"Deployment {deployment_id} rolled back successfully")
                return success
        except Exception as e:
            logger.error(f"Error rolling back deployment {deployment_id}: {e}")
            return False
    
    def _get_accuracy_chart_data(self, hours_back: int = 24) -> Dict[str, Any]:
        """Get accuracy trend chart data"""
        with DatabaseConnection(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT mp.model_id, mr.model_name, mp.timestamp, mp.accuracy_score
                FROM model_performance mp
                JOIN model_registry mr ON mp.model_id = mr.model_id
                WHERE mp.timestamp > datetime('now', '-{} hours')
                ORDER BY mp.timestamp ASC
            """.format(hours_back))
            
            data = cursor.fetchall()
            
            # Group by model
            model_data = {}
            for model_id, model_name, timestamp, accuracy in data:
                if model_id not in model_data:
                    model_data[model_id] = {
                        'name': model_name,
                        'timestamps': [],
                        'accuracy': []
                    }
                model_data[model_id]['timestamps'].append(timestamp)
                model_data[model_id]['accuracy'].append(accuracy)
            
            # Create Plotly traces
            traces = []
            for model_id, data in model_data.items():
                traces.append({
                    'x': data['timestamps'],
                    'y': data['accuracy'],
                    'type': 'scatter',
                    'mode': 'lines+markers',
                    'name': data['name'],
                    'line': {'width': 2}
                })
            
            layout = {
                'title': f'Model Accuracy Trends (Last {hours_back} Hours)',
                'xaxis': {'title': 'Time'},
                'yaxis': {'title': 'Accuracy Score', 'range': [0, 1]},
                'hovermode': 'x unified'
            }
            
            return {
                'data': traces,
                'layout': layout
            }
    
    def _get_drift_chart_data(self, days_back: int = 7) -> Dict[str, Any]:
        """Get drift detection chart data"""
        with DatabaseConnection(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT DATE(detected_at) as date, severity, COUNT(*) as count
                FROM drift_events
                WHERE detected_at > datetime('now', '-{} days')
                GROUP BY DATE(detected_at), severity
                ORDER BY date ASC
            """.format(days_back))
            
            data = cursor.fetchall()
            
            # Group by severity
            severity_data = {'LOW': [], 'MEDIUM': [], 'HIGH': []}
            dates = []
            
            # Get all dates in range
            cursor.execute("""
                SELECT DISTINCT DATE(detected_at) as date
                FROM drift_events
                WHERE detected_at > datetime('now', '-{} days')
                ORDER BY date ASC
            """.format(days_back))
            
            all_dates = [row[0] for row in cursor.fetchall()]
            
            # Fill data
            for date in all_dates:
                dates.append(date)
                for severity in ['LOW', 'MEDIUM', 'HIGH']:
                    count = next((row[2] for row in data if row[0] == date and row[1] == severity), 0)
                    severity_data[severity].append(count)
            
            # Create Plotly traces
            traces = []
            colors = {'LOW': '#28a745', 'MEDIUM': '#ffc107', 'HIGH': '#dc3545'}
            
            for severity in ['LOW', 'MEDIUM', 'HIGH']:
                traces.append({
                    'x': dates,
                    'y': severity_data[severity],
                    'type': 'bar',
                    'name': f'{severity} Severity',
                    'marker': {'color': colors[severity]}
                })
            
            layout = {
                'title': f'Drift Events by Severity (Last {days_back} Days)',
                'xaxis': {'title': 'Date'},
                'yaxis': {'title': 'Number of Events'},
                'barmode': 'stack'
            }
            
            return {
                'data': traces,
                'layout': layout
            }
    
    def _broadcast_updates(self):
        """Broadcast real-time updates to connected clients"""
        try:
            # Get current metrics
            metrics = self._get_dashboard_metrics()
            self.socketio.emit('metrics_update', asdict(metrics))
            
            # Get model performance
            models = self._get_model_performance_data()
            self.socketio.emit('models_update', [asdict(model) for model in models])
            
            # Get alerts
            alerts = self._get_current_alerts()
            self.socketio.emit('alerts_update', [asdict(alert) for alert in alerts])
            
        except Exception as e:
            logger.error(f"Error broadcasting updates: {e}")
            self.socketio.emit('error', {'msg': str(e)})
    
    def _real_time_update_loop(self):
        """Real-time update loop for dashboard"""
        while self.running:
            try:
                self._broadcast_updates()
                time.sleep(30)  # Update every 30 seconds
            except Exception as e:
                logger.error(f"Error in real-time update loop: {e}")
                time.sleep(60)  # Wait longer on error
    
    def start_dashboard(self, host: str = '0.0.0.0', port: int = 5000, debug: bool = False):
        """Start the monitoring dashboard"""
        logger.info(f"Starting monitoring dashboard on {host}:{port}")
        
        # Start real-time update thread
        self.running = True
        self.update_thread = threading.Thread(target=self._real_time_update_loop, daemon=True)
        self.update_thread.start()
        
        # Start Flask-SocketIO server
        self.socketio.run(self.app, host=host, port=port, debug=debug)
    
    def stop_dashboard(self):
        """Stop the monitoring dashboard"""
        logger.info("Stopping monitoring dashboard")
        self.running = False
        if self.update_thread:
            self.update_thread.join(timeout=5)

# Dashboard instance for external access
dashboard = MonitoringDashboard()

if __name__ == "__main__":
    dashboard.start_dashboard(debug=True)