"""
Alerting and Notification System
Handles alert generation, severity classification, and multi-channel notifications.
"""

import json
import smtplib
import requests
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import threading
import time
import queue
import sqlite3

from .database.connection import DatabaseConnection
from .database.utils import RetailDatabaseUtils

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    """Alert severity levels"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class AlertType(Enum):
    """Types of alerts"""
    MODEL_DRIFT = "MODEL_DRIFT"
    ACCURACY_DROP = "ACCURACY_DROP"
    DEPLOYMENT_FAILURE = "DEPLOYMENT_FAILURE"
    RETRAINING_FAILURE = "RETRAINING_FAILURE"
    SYSTEM_ERROR = "SYSTEM_ERROR"
    BUSINESS_IMPACT = "BUSINESS_IMPACT"

class NotificationChannel(Enum):
    """Notification delivery channels"""
    EMAIL = "EMAIL"
    SLACK = "SLACK"
    DASHBOARD = "DASHBOARD"
    SMS = "SMS"
    WEBHOOK = "WEBHOOK"

class AlertStatus(Enum):
    """Alert status"""
    ACTIVE = "ACTIVE"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    RESOLVED = "RESOLVED"
    ESCALATED = "ESCALATED"

@dataclass
class Alert:
    """Alert data model"""
    alert_id: str
    alert_type: AlertType
    severity: AlertSeverity
    title: str
    message: str
    model_id: Optional[str]
    category: Optional[str]
    timestamp: datetime
    status: AlertStatus
    metadata: Dict[str, Any]
    escalation_level: int = 0
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None

@dataclass
class NotificationConfig:
    """Notification configuration"""
    channel: NotificationChannel
    enabled: bool
    config: Dict[str, Any]
    severity_filter: List[AlertSeverity]

@dataclass
class EscalationRule:
    """Alert escalation rule"""
    severity: AlertSeverity
    escalation_delay_minutes: int
    max_escalation_level: int
    escalation_channels: List[NotificationChannel]

class AlertGenerator:
    """Generates alerts based on system conditions"""
    
    def __init__(self, db_path: str = "autonomous_demand_forecasting.db"):
        self.db_path = db_path
        self.db_utils = RetailDatabaseUtils(db_path)
        
        # Alert thresholds
        self.accuracy_thresholds = {
            AlertSeverity.LOW: 0.85,
            AlertSeverity.MEDIUM: 0.80,
            AlertSeverity.HIGH: 0.75,
            AlertSeverity.CRITICAL: 0.70
        }
        
        self.drift_thresholds = {
            AlertSeverity.LOW: 0.05,
            AlertSeverity.MEDIUM: 0.10,
            AlertSeverity.HIGH: 0.15,
            AlertSeverity.CRITICAL: 0.20
        }
    
    def generate_model_drift_alert(self, model_id: str, accuracy_drop: float, 
                                 affected_categories: List[str]) -> Alert:
        """Generate model drift alert"""
        severity = self._classify_drift_severity(accuracy_drop)
        
        alert_id = f"drift_{model_id}_{int(datetime.now().timestamp())}"
        
        title = f"Model Drift Detected - {model_id}"
        message = (f"Model {model_id} has experienced a {accuracy_drop:.1%} accuracy drop. "
                  f"Affected categories: {', '.join(affected_categories)}")
        
        metadata = {
            "accuracy_drop": accuracy_drop,
            "affected_categories": affected_categories,
            "threshold_exceeded": accuracy_drop > self.drift_thresholds[AlertSeverity.LOW]
        }
        
        return Alert(
            alert_id=alert_id,
            alert_type=AlertType.MODEL_DRIFT,
            severity=severity,
            title=title,
            message=message,
            model_id=model_id,
            category=affected_categories[0] if affected_categories else None,
            timestamp=datetime.now(),
            status=AlertStatus.ACTIVE,
            metadata=metadata
        )
    
    def generate_accuracy_drop_alert(self, model_id: str, current_accuracy: float, 
                                   previous_accuracy: float, category: str) -> Alert:
        """Generate accuracy drop alert"""
        accuracy_drop = previous_accuracy - current_accuracy
        severity = self._classify_accuracy_severity(current_accuracy)
        
        alert_id = f"accuracy_{model_id}_{category}_{int(datetime.now().timestamp())}"
        
        title = f"Accuracy Drop Alert - {model_id}"
        message = (f"Model {model_id} accuracy in {category} dropped from "
                  f"{previous_accuracy:.1%} to {current_accuracy:.1%}")
        
        metadata = {
            "current_accuracy": current_accuracy,
            "previous_accuracy": previous_accuracy,
            "accuracy_drop": accuracy_drop,
            "category": category
        }
        
        return Alert(
            alert_id=alert_id,
            alert_type=AlertType.ACCURACY_DROP,
            severity=severity,
            title=title,
            message=message,
            model_id=model_id,
            category=category,
            timestamp=datetime.now(),
            status=AlertStatus.ACTIVE,
            metadata=metadata
        )
    
    def generate_deployment_failure_alert(self, deployment_id: str, model_id: str, 
                                        error_message: str) -> Alert:
        """Generate deployment failure alert"""
        alert_id = f"deploy_fail_{deployment_id}_{int(datetime.now().timestamp())}"
        
        title = f"Deployment Failure - {model_id}"
        message = f"Deployment {deployment_id} for model {model_id} failed: {error_message}"
        
        metadata = {
            "deployment_id": deployment_id,
            "error_message": error_message,
            "failure_time": datetime.now().isoformat()
        }
        
        return Alert(
            alert_id=alert_id,
            alert_type=AlertType.DEPLOYMENT_FAILURE,
            severity=AlertSeverity.HIGH,
            title=title,
            message=message,
            model_id=model_id,
            category=None,
            timestamp=datetime.now(),
            status=AlertStatus.ACTIVE,
            metadata=metadata
        )
    
    def generate_retraining_failure_alert(self, workflow_id: str, model_id: str, 
                                        error_message: str) -> Alert:
        """Generate retraining failure alert"""
        alert_id = f"retrain_fail_{workflow_id}_{int(datetime.now().timestamp())}"
        
        title = f"Retraining Failure - {model_id}"
        message = f"Retraining workflow {workflow_id} for model {model_id} failed: {error_message}"
        
        metadata = {
            "workflow_id": workflow_id,
            "error_message": error_message,
            "failure_time": datetime.now().isoformat()
        }
        
        return Alert(
            alert_id=alert_id,
            alert_type=AlertType.RETRAINING_FAILURE,
            severity=AlertSeverity.MEDIUM,
            title=title,
            message=message,
            model_id=model_id,
            category=None,
            timestamp=datetime.now(),
            status=AlertStatus.ACTIVE,
            metadata=metadata
        )
    
    def generate_business_impact_alert(self, impact_type: str, impact_value: float, 
                                     threshold: float, model_id: str) -> Alert:
        """Generate business impact alert"""
        severity = AlertSeverity.HIGH if impact_value < threshold * 0.5 else AlertSeverity.MEDIUM
        
        alert_id = f"business_{impact_type}_{model_id}_{int(datetime.now().timestamp())}"
        
        title = f"Business Impact Alert - {impact_type}"
        message = (f"Business impact metric '{impact_type}' for model {model_id} "
                  f"is {impact_value:.2f}, below threshold of {threshold:.2f}")
        
        metadata = {
            "impact_type": impact_type,
            "impact_value": impact_value,
            "threshold": threshold,
            "deviation": threshold - impact_value
        }
        
        return Alert(
            alert_id=alert_id,
            alert_type=AlertType.BUSINESS_IMPACT,
            severity=severity,
            title=title,
            message=message,
            model_id=model_id,
            category=None,
            timestamp=datetime.now(),
            status=AlertStatus.ACTIVE,
            metadata=metadata
        )
    
    def _classify_drift_severity(self, accuracy_drop: float) -> AlertSeverity:
        """Classify drift severity based on accuracy drop"""
        if accuracy_drop >= self.drift_thresholds[AlertSeverity.CRITICAL]:
            return AlertSeverity.CRITICAL
        elif accuracy_drop >= self.drift_thresholds[AlertSeverity.HIGH]:
            return AlertSeverity.HIGH
        elif accuracy_drop >= self.drift_thresholds[AlertSeverity.MEDIUM]:
            return AlertSeverity.MEDIUM
        else:
            return AlertSeverity.LOW
    
    def _classify_accuracy_severity(self, accuracy: float) -> AlertSeverity:
        """Classify accuracy severity based on current accuracy"""
        if accuracy < self.accuracy_thresholds[AlertSeverity.CRITICAL]:
            return AlertSeverity.CRITICAL
        elif accuracy < self.accuracy_thresholds[AlertSeverity.HIGH]:
            return AlertSeverity.HIGH
        elif accuracy < self.accuracy_thresholds[AlertSeverity.MEDIUM]:
            return AlertSeverity.MEDIUM
        else:
            return AlertSeverity.LOW

class NotificationDelivery:
    """Handles multi-channel notification delivery"""
    
    def __init__(self, config: Dict[str, NotificationConfig]):
        self.config = config
        self.delivery_queue = queue.Queue()
        self.delivery_thread = None
        self.running = False
    
    def send_email_notification(self, alert: Alert, email_config: Dict[str, Any]) -> bool:
        """Send email notification"""
        try:
            smtp_server = email_config.get('smtp_server')
            smtp_port = email_config.get('smtp_port', 587)
            username = email_config.get('username')
            password = email_config.get('password')
            recipients = email_config.get('recipients', [])
            
            if not all([smtp_server, username, password, recipients]):
                logger.error("Incomplete email configuration")
                return False
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = username
            msg['To'] = ', '.join(recipients)
            msg['Subject'] = f"[{alert.severity.value}] {alert.title}"
            
            # Email body
            body = f"""
Alert Details:
- Alert ID: {alert.alert_id}
- Type: {alert.alert_type.value}
- Severity: {alert.severity.value}
- Model: {alert.model_id or 'N/A'}
- Category: {alert.category or 'N/A'}
- Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}

Message:
{alert.message}

Metadata:
{json.dumps(alert.metadata, indent=2)}

Please acknowledge this alert in the monitoring dashboard.
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            server.login(username, password)
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Email notification sent for alert {alert.alert_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")
            return False
    
    def send_slack_notification(self, alert: Alert, slack_config: Dict[str, Any]) -> bool:
        """Send Slack notification"""
        try:
            webhook_url = slack_config.get('webhook_url')
            channel = slack_config.get('channel', '#alerts')
            
            if not webhook_url:
                logger.error("Slack webhook URL not configured")
                return False
            
            # Determine color based on severity
            color_map = {
                AlertSeverity.LOW: "#36a64f",      # Green
                AlertSeverity.MEDIUM: "#ff9500",   # Orange
                AlertSeverity.HIGH: "#ff0000",     # Red
                AlertSeverity.CRITICAL: "#8B0000"  # Dark Red
            }
            
            # Create Slack message
            payload = {
                "channel": channel,
                "username": "Demand Forecasting Bot",
                "icon_emoji": ":warning:",
                "attachments": [
                    {
                        "color": color_map.get(alert.severity, "#ff0000"),
                        "title": alert.title,
                        "text": alert.message,
                        "fields": [
                            {
                                "title": "Alert ID",
                                "value": alert.alert_id,
                                "short": True
                            },
                            {
                                "title": "Severity",
                                "value": alert.severity.value,
                                "short": True
                            },
                            {
                                "title": "Model",
                                "value": alert.model_id or "N/A",
                                "short": True
                            },
                            {
                                "title": "Category",
                                "value": alert.category or "N/A",
                                "short": True
                            },
                            {
                                "title": "Time",
                                "value": alert.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                                "short": False
                            }
                        ],
                        "footer": "Autonomous Demand Forecasting",
                        "ts": int(alert.timestamp.timestamp())
                    }
                ]
            }
            
            # Send to Slack
            response = requests.post(webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            
            logger.info(f"Slack notification sent for alert {alert.alert_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send Slack notification: {e}")
            return False
    
    def send_webhook_notification(self, alert: Alert, webhook_config: Dict[str, Any]) -> bool:
        """Send webhook notification"""
        try:
            url = webhook_config.get('url')
            headers = webhook_config.get('headers', {})
            
            if not url:
                logger.error("Webhook URL not configured")
                return False
            
            # Create webhook payload
            payload = {
                "alert": asdict(alert),
                "timestamp": alert.timestamp.isoformat(),
                "source": "autonomous_demand_forecasting"
            }
            
            # Convert enums to strings for JSON serialization
            payload['alert']['alert_type'] = alert.alert_type.value
            payload['alert']['severity'] = alert.severity.value
            payload['alert']['status'] = alert.status.value
            
            # Send webhook
            response = requests.post(url, json=payload, headers=headers, timeout=10)
            response.raise_for_status()
            
            logger.info(f"Webhook notification sent for alert {alert.alert_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send webhook notification: {e}")
            return False
    
    def deliver_notification(self, alert: Alert, channel: NotificationChannel) -> bool:
        """Deliver notification via specified channel"""
        if channel not in self.config or not self.config[channel].enabled:
            logger.warning(f"Channel {channel.value} not configured or disabled")
            return False
        
        config = self.config[channel]
        
        # Check severity filter
        if config.severity_filter and alert.severity not in config.severity_filter:
            logger.debug(f"Alert severity {alert.severity.value} filtered out for channel {channel.value}")
            return False
        
        # Route to appropriate delivery method
        if channel == NotificationChannel.EMAIL:
            return self.send_email_notification(alert, config.config)
        elif channel == NotificationChannel.SLACK:
            return self.send_slack_notification(alert, config.config)
        elif channel == NotificationChannel.WEBHOOK:
            return self.send_webhook_notification(alert, config.config)
        elif channel == NotificationChannel.DASHBOARD:
            # Dashboard notifications are handled by the dashboard itself
            return True
        else:
            logger.warning(f"Unsupported notification channel: {channel.value}")
            return False
    
    def start_delivery_service(self):
        """Start the notification delivery service"""
        self.running = True
        self.delivery_thread = threading.Thread(target=self._delivery_worker, daemon=True)
        self.delivery_thread.start()
        logger.info("Notification delivery service started")
    
    def stop_delivery_service(self):
        """Stop the notification delivery service"""
        self.running = False
        if self.delivery_thread:
            self.delivery_thread.join(timeout=5)
        logger.info("Notification delivery service stopped")
    
    def _delivery_worker(self):
        """Background worker for notification delivery"""
        while self.running:
            try:
                # Get notification from queue (with timeout)
                alert, channels = self.delivery_queue.get(timeout=1)
                
                # Deliver to all specified channels
                for channel in channels:
                    try:
                        self.deliver_notification(alert, channel)
                    except Exception as e:
                        logger.error(f"Failed to deliver notification via {channel.value}: {e}")
                
                self.delivery_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in notification delivery worker: {e}")
    
    def queue_notification(self, alert: Alert, channels: List[NotificationChannel]):
        """Queue notification for delivery"""
        self.delivery_queue.put((alert, channels))

class AlertEscalationManager:
    """Manages alert escalation workflows"""
    
    def __init__(self, escalation_rules: List[EscalationRule], 
                 notification_delivery: NotificationDelivery):
        self.escalation_rules = {rule.severity: rule for rule in escalation_rules}
        self.notification_delivery = notification_delivery
        self.escalation_thread = None
        self.running = False
        self.active_escalations = {}
    
    def start_escalation_service(self):
        """Start the escalation service"""
        self.running = True
        self.escalation_thread = threading.Thread(target=self._escalation_worker, daemon=True)
        self.escalation_thread.start()
        logger.info("Alert escalation service started")
    
    def stop_escalation_service(self):
        """Stop the escalation service"""
        self.running = False
        if self.escalation_thread:
            self.escalation_thread.join(timeout=5)
        logger.info("Alert escalation service stopped")
    
    def register_alert_for_escalation(self, alert: Alert):
        """Register alert for potential escalation"""
        if alert.severity in self.escalation_rules:
            rule = self.escalation_rules[alert.severity]
            escalation_time = alert.timestamp + timedelta(minutes=rule.escalation_delay_minutes)
            
            self.active_escalations[alert.alert_id] = {
                'alert': alert,
                'rule': rule,
                'escalation_time': escalation_time,
                'current_level': 0
            }
            
            logger.info(f"Alert {alert.alert_id} registered for escalation at {escalation_time}")
    
    def cancel_escalation(self, alert_id: str):
        """Cancel escalation for resolved/acknowledged alert"""
        if alert_id in self.active_escalations:
            del self.active_escalations[alert_id]
            logger.info(f"Escalation cancelled for alert {alert_id}")
    
    def _escalation_worker(self):
        """Background worker for alert escalation"""
        while self.running:
            try:
                current_time = datetime.now()
                escalations_to_process = []
                
                # Find alerts ready for escalation
                for alert_id, escalation_data in self.active_escalations.items():
                    if current_time >= escalation_data['escalation_time']:
                        escalations_to_process.append(alert_id)
                
                # Process escalations
                for alert_id in escalations_to_process:
                    self._process_escalation(alert_id)
                
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in escalation worker: {e}")
    
    def _process_escalation(self, alert_id: str):
        """Process alert escalation"""
        escalation_data = self.active_escalations[alert_id]
        alert = escalation_data['alert']
        rule = escalation_data['rule']
        current_level = escalation_data['current_level']
        
        # Check if we've reached max escalation level
        if current_level >= rule.max_escalation_level:
            logger.warning(f"Alert {alert_id} reached maximum escalation level")
            return
        
        # Escalate alert
        escalated_alert = Alert(
            alert_id=f"{alert.alert_id}_escalated_{current_level + 1}",
            alert_type=alert.alert_type,
            severity=alert.severity,
            title=f"[ESCALATED] {alert.title}",
            message=f"ESCALATED ALERT (Level {current_level + 1}): {alert.message}",
            model_id=alert.model_id,
            category=alert.category,
            timestamp=datetime.now(),
            status=AlertStatus.ESCALATED,
            metadata={**alert.metadata, "escalation_level": current_level + 1, "original_alert_id": alert.alert_id},
            escalation_level=current_level + 1
        )
        
        # Send escalated notification
        self.notification_delivery.queue_notification(escalated_alert, rule.escalation_channels)
        
        # Update escalation data
        escalation_data['current_level'] = current_level + 1
        escalation_data['escalation_time'] = datetime.now() + timedelta(minutes=rule.escalation_delay_minutes)
        
        logger.info(f"Alert {alert_id} escalated to level {current_level + 1}")

class AlertingNotificationSystem:
    """Main alerting and notification system"""
    
    def __init__(self, db_path: str = "autonomous_demand_forecasting.db",
                 notification_config: Optional[Dict[str, NotificationConfig]] = None,
                 escalation_rules: Optional[List[EscalationRule]] = None):
        self.db_path = db_path
        self.alert_generator = AlertGenerator(db_path)
        
        # Default notification configuration
        if notification_config is None:
            notification_config = self._get_default_notification_config()
        
        self.notification_delivery = NotificationDelivery(notification_config)
        
        # Default escalation rules
        if escalation_rules is None:
            escalation_rules = self._get_default_escalation_rules()
        
        self.escalation_manager = AlertEscalationManager(escalation_rules, self.notification_delivery)
        
        # Alert storage
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        
        # Initialize database tables
        self._initialize_alert_tables()
    
    def _get_default_notification_config(self) -> Dict[str, NotificationConfig]:
        """Get default notification configuration"""
        return {
            NotificationChannel.EMAIL: NotificationConfig(
                channel=NotificationChannel.EMAIL,
                enabled=False,  # Disabled by default, requires configuration
                config={},
                severity_filter=[AlertSeverity.MEDIUM, AlertSeverity.HIGH, AlertSeverity.CRITICAL]
            ),
            NotificationChannel.SLACK: NotificationConfig(
                channel=NotificationChannel.SLACK,
                enabled=False,  # Disabled by default, requires configuration
                config={},
                severity_filter=[AlertSeverity.HIGH, AlertSeverity.CRITICAL]
            ),
            NotificationChannel.DASHBOARD: NotificationConfig(
                channel=NotificationChannel.DASHBOARD,
                enabled=True,
                config={},
                severity_filter=list(AlertSeverity)  # All severities
            ),
            NotificationChannel.WEBHOOK: NotificationConfig(
                channel=NotificationChannel.WEBHOOK,
                enabled=False,  # Disabled by default, requires configuration
                config={},
                severity_filter=[AlertSeverity.CRITICAL]
            )
        }
    
    def _get_default_escalation_rules(self) -> List[EscalationRule]:
        """Get default escalation rules"""
        return [
            EscalationRule(
                severity=AlertSeverity.CRITICAL,
                escalation_delay_minutes=15,
                max_escalation_level=3,
                escalation_channels=[NotificationChannel.EMAIL, NotificationChannel.SLACK]
            ),
            EscalationRule(
                severity=AlertSeverity.HIGH,
                escalation_delay_minutes=30,
                max_escalation_level=2,
                escalation_channels=[NotificationChannel.EMAIL]
            ),
            EscalationRule(
                severity=AlertSeverity.MEDIUM,
                escalation_delay_minutes=60,
                max_escalation_level=1,
                escalation_channels=[NotificationChannel.EMAIL]
            )
        ]
    
    def _initialize_alert_tables(self):
        """Initialize database tables for alert storage"""
        db_conn = DatabaseConnection(self.db_path)
        with db_conn.transaction() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY,
                    alert_id TEXT UNIQUE NOT NULL,
                    alert_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    title TEXT NOT NULL,
                    message TEXT NOT NULL,
                    model_id TEXT,
                    category TEXT,
                    timestamp DATETIME NOT NULL,
                    status TEXT NOT NULL,
                    metadata TEXT,
                    escalation_level INTEGER DEFAULT 0,
                    acknowledged_by TEXT,
                    acknowledged_at DATETIME,
                    resolved_at DATETIME,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS alert_notifications (
                    id INTEGER PRIMARY KEY,
                    alert_id TEXT NOT NULL,
                    channel TEXT NOT NULL,
                    status TEXT NOT NULL,
                    sent_at DATETIME NOT NULL,
                    error_message TEXT,
                    FOREIGN KEY (alert_id) REFERENCES alerts(alert_id)
                )
            """)
            
            # Transaction will auto-commit
    
    def start_services(self):
        """Start all alerting services"""
        self.notification_delivery.start_delivery_service()
        self.escalation_manager.start_escalation_service()
        logger.info("Alerting and notification system started")
    
    def stop_services(self):
        """Stop all alerting services"""
        self.notification_delivery.stop_delivery_service()
        self.escalation_manager.stop_escalation_service()
        logger.info("Alerting and notification system stopped")
    
    def create_alert(self, alert: Alert, channels: Optional[List[NotificationChannel]] = None) -> str:
        """Create and process a new alert"""
        # Store alert
        self.active_alerts[alert.alert_id] = alert
        self.alert_history.append(alert)
        
        # Save to database
        self._save_alert_to_database(alert)
        
        # Determine notification channels
        if channels is None:
            channels = self._get_default_channels_for_severity(alert.severity)
        
        # Queue notifications
        self.notification_delivery.queue_notification(alert, channels)
        
        # Register for escalation if needed
        self.escalation_manager.register_alert_for_escalation(alert)
        
        logger.info(f"Alert created: {alert.alert_id} ({alert.severity.value})")
        return alert.alert_id
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Acknowledge an alert"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.ACKNOWLEDGED
            alert.acknowledged_by = acknowledged_by
            alert.acknowledged_at = datetime.now()
            
            # Update database
            self._update_alert_in_database(alert)
            
            # Cancel escalation
            self.escalation_manager.cancel_escalation(alert_id)
            
            logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")
            return True
        
        return False
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.RESOLVED
            alert.resolved_at = datetime.now()
            
            # Update database
            self._update_alert_in_database(alert)
            
            # Cancel escalation
            self.escalation_manager.cancel_escalation(alert_id)
            
            # Remove from active alerts
            del self.active_alerts[alert_id]
            
            logger.info(f"Alert {alert_id} resolved")
            return True
        
        return False
    
    def get_active_alerts(self, severity_filter: Optional[List[AlertSeverity]] = None) -> List[Alert]:
        """Get active alerts with optional severity filter"""
        alerts = list(self.active_alerts.values())
        
        if severity_filter:
            alerts = [alert for alert in alerts if alert.severity in severity_filter]
        
        return sorted(alerts, key=lambda x: x.timestamp, reverse=True)
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alert statistics"""
        active_alerts = list(self.active_alerts.values())
        
        stats = {
            "total_active": len(active_alerts),
            "by_severity": {severity.value: 0 for severity in AlertSeverity},
            "by_type": {alert_type.value: 0 for alert_type in AlertType},
            "by_status": {status.value: 0 for status in AlertStatus},
            "escalated_count": 0
        }
        
        for alert in active_alerts:
            stats["by_severity"][alert.severity.value] += 1
            stats["by_type"][alert.alert_type.value] += 1
            stats["by_status"][alert.status.value] += 1
            
            if alert.escalation_level > 0:
                stats["escalated_count"] += 1
        
        return stats
    
    def _get_default_channels_for_severity(self, severity: AlertSeverity) -> List[NotificationChannel]:
        """Get default notification channels for severity level"""
        if severity == AlertSeverity.CRITICAL:
            return [NotificationChannel.EMAIL, NotificationChannel.SLACK, NotificationChannel.DASHBOARD]
        elif severity == AlertSeverity.HIGH:
            return [NotificationChannel.EMAIL, NotificationChannel.DASHBOARD]
        elif severity == AlertSeverity.MEDIUM:
            return [NotificationChannel.DASHBOARD]
        else:
            return [NotificationChannel.DASHBOARD]
    
    def _save_alert_to_database(self, alert: Alert):
        """Save alert to database"""
        db_conn = DatabaseConnection(self.db_path)
        with db_conn.transaction() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO alerts 
                (alert_id, alert_type, severity, title, message, model_id, category,
                 timestamp, status, metadata, escalation_level, acknowledged_by, 
                 acknowledged_at, resolved_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                alert.alert_id, alert.alert_type.value, alert.severity.value,
                alert.title, alert.message, alert.model_id, alert.category,
                alert.timestamp, alert.status.value, json.dumps(alert.metadata),
                alert.escalation_level, alert.acknowledged_by,
                alert.acknowledged_at, alert.resolved_at
            ))
            
            # Transaction will auto-commit
    
    def _update_alert_in_database(self, alert: Alert):
        """Update alert in database"""
        db_conn = DatabaseConnection(self.db_path)
        with db_conn.transaction() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE alerts 
                SET status = ?, acknowledged_by = ?, acknowledged_at = ?, resolved_at = ?
                WHERE alert_id = ?
            """, (
                alert.status.value, alert.acknowledged_by,
                alert.acknowledged_at, alert.resolved_at, alert.alert_id
            ))
            
            # Transaction will auto-commit

# Global alerting system instance
_alerting_system: Optional[AlertingNotificationSystem] = None

def get_alerting_system(db_path: str = "autonomous_demand_forecasting.db") -> AlertingNotificationSystem:
    """Get or create global alerting system instance"""
    global _alerting_system
    if _alerting_system is None:
        _alerting_system = AlertingNotificationSystem(db_path)
    return _alerting_system