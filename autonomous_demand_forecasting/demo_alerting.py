#!/usr/bin/env python3
"""
Demo script for the Alerting and Notification System
Shows the alerting system with sample alerts and notifications
"""

import os
import time
from datetime import datetime, timedelta
from alerting_notification_system import (
    AlertingNotificationSystem, AlertSeverity, AlertType, NotificationChannel, 
    NotificationConfig, EscalationRule
)

def setup_demo_alerting_system():
    """Set up demo alerting system with custom configuration"""
    
    # Custom notification configuration for demo
    notification_config = {
        NotificationChannel.EMAIL: NotificationConfig(
            channel=NotificationChannel.EMAIL,
            enabled=False,  # Disabled for demo (no real SMTP server)
            config={
                'smtp_server': 'smtp.example.com',
                'smtp_port': 587,
                'username': 'alerts@company.com',
                'password': 'demo_password',
                'recipients': ['admin@company.com', 'ops@company.com']
            },
            severity_filter=[AlertSeverity.MEDIUM, AlertSeverity.HIGH, AlertSeverity.CRITICAL]
        ),
        NotificationChannel.SLACK: NotificationConfig(
            channel=NotificationChannel.SLACK,
            enabled=False,  # Disabled for demo (no real webhook)
            config={
                'webhook_url': 'https://hooks.slack.com/services/demo/webhook',
                'channel': '#alerts'
            },
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
            enabled=False,  # Disabled for demo
            config={
                'url': 'https://api.company.com/alerts',
                'headers': {'Authorization': 'Bearer demo_token'}
            },
            severity_filter=[AlertSeverity.CRITICAL]
        )
    }
    
    # Custom escalation rules for demo
    escalation_rules = [
        EscalationRule(
            severity=AlertSeverity.CRITICAL,
            escalation_delay_minutes=5,  # Short delay for demo
            max_escalation_level=3,
            escalation_channels=[NotificationChannel.EMAIL, NotificationChannel.SLACK]
        ),
        EscalationRule(
            severity=AlertSeverity.HIGH,
            escalation_delay_minutes=10,  # Short delay for demo
            max_escalation_level=2,
            escalation_channels=[NotificationChannel.EMAIL]
        ),
        EscalationRule(
            severity=AlertSeverity.MEDIUM,
            escalation_delay_minutes=15,  # Short delay for demo
            max_escalation_level=1,
            escalation_channels=[NotificationChannel.EMAIL]
        )
    ]
    
    # Create alerting system
    demo_db_path = "demo_alerting.db"
    alerting_system = AlertingNotificationSystem(
        db_path=demo_db_path,
        notification_config=notification_config,
        escalation_rules=escalation_rules
    )
    
    return alerting_system, demo_db_path

def generate_sample_alerts(alerting_system):
    """Generate various sample alerts for demonstration"""
    
    print("📢 Generating sample alerts...")
    
    # 1. Model drift alert (MEDIUM severity)
    drift_alert = alerting_system.alert_generator.generate_model_drift_alert(
        model_id="electronics_forecast_v1",
        accuracy_drop=0.12,
        affected_categories=["electronics", "computers"]
    )
    drift_alert_id = alerting_system.create_alert(drift_alert)
    print(f"   ✓ Model drift alert created: {drift_alert_id}")
    
    # 2. Accuracy drop alert (HIGH severity)
    accuracy_alert = alerting_system.alert_generator.generate_accuracy_drop_alert(
        model_id="clothing_forecast_v1",
        current_accuracy=0.72,
        previous_accuracy=0.88,
        category="clothing"
    )
    accuracy_alert_id = alerting_system.create_alert(accuracy_alert)
    print(f"   ✓ Accuracy drop alert created: {accuracy_alert_id}")
    
    # 3. Deployment failure alert (HIGH severity)
    deployment_alert = alerting_system.alert_generator.generate_deployment_failure_alert(
        deployment_id="deploy_books_v2_001",
        model_id="books_forecast_v2",
        error_message="Model validation failed - accuracy below threshold"
    )
    deployment_alert_id = alerting_system.create_alert(deployment_alert)
    print(f"   ✓ Deployment failure alert created: {deployment_alert_id}")
    
    # 4. Retraining failure alert (MEDIUM severity)
    retraining_alert = alerting_system.alert_generator.generate_retraining_failure_alert(
        workflow_id="retrain_home_garden_003",
        model_id="home_garden_forecast_v1",
        error_message="Insufficient training data - only 30 days available"
    )
    retraining_alert_id = alerting_system.create_alert(retraining_alert)
    print(f"   ✓ Retraining failure alert created: {retraining_alert_id}")
    
    # 5. Business impact alert (HIGH severity)
    business_alert = alerting_system.alert_generator.generate_business_impact_alert(
        impact_type="revenue_loss",
        impact_value=25000.0,
        threshold=50000.0,
        model_id="electronics_forecast_v1"
    )
    business_alert_id = alerting_system.create_alert(business_alert)
    print(f"   ✓ Business impact alert created: {business_alert_id}")
    
    # 6. Critical system error (CRITICAL severity)
    critical_alert = alerting_system.alert_generator.generate_model_drift_alert(
        model_id="critical_model_v1",
        accuracy_drop=0.35,  # Very high drop
        affected_categories=["all_categories"]
    )
    critical_alert_id = alerting_system.create_alert(critical_alert)
    print(f"   ✓ Critical alert created: {critical_alert_id}")
    
    return [drift_alert_id, accuracy_alert_id, deployment_alert_id, 
            retraining_alert_id, business_alert_id, critical_alert_id]

def demonstrate_alert_management(alerting_system, alert_ids):
    """Demonstrate alert management operations"""
    
    print("\n🔧 Demonstrating alert management...")
    
    # Show active alerts
    active_alerts = alerting_system.get_active_alerts()
    print(f"   📊 Total active alerts: {len(active_alerts)}")
    
    # Show alert statistics
    stats = alerting_system.get_alert_statistics()
    print(f"   📈 Alert statistics:")
    print(f"      - Total active: {stats['total_active']}")
    print(f"      - By severity: {stats['by_severity']}")
    print(f"      - By type: {stats['by_type']}")
    print(f"      - Escalated: {stats['escalated_count']}")
    
    # Acknowledge some alerts
    print(f"\n   ✅ Acknowledging alerts...")
    alerting_system.acknowledge_alert(alert_ids[0], "admin_user")
    alerting_system.acknowledge_alert(alert_ids[2], "ops_team")
    print(f"      - Acknowledged 2 alerts")
    
    # Resolve some alerts
    print(f"   ✅ Resolving alerts...")
    alerting_system.resolve_alert(alert_ids[0])
    alerting_system.resolve_alert(alert_ids[3])
    print(f"      - Resolved 2 alerts")
    
    # Show updated statistics
    updated_stats = alerting_system.get_alert_statistics()
    print(f"   📈 Updated statistics:")
    print(f"      - Total active: {updated_stats['total_active']}")
    print(f"      - By status: {updated_stats['by_status']}")

def demonstrate_severity_filtering(alerting_system):
    """Demonstrate severity-based alert filtering"""
    
    print("\n🔍 Demonstrating severity filtering...")
    
    # Get high and critical alerts only
    high_critical_alerts = alerting_system.get_active_alerts(
        severity_filter=[AlertSeverity.HIGH, AlertSeverity.CRITICAL]
    )
    print(f"   🚨 High/Critical alerts: {len(high_critical_alerts)}")
    
    for alert in high_critical_alerts:
        print(f"      - {alert.severity.value}: {alert.title}")
    
    # Get medium alerts only
    medium_alerts = alerting_system.get_active_alerts(
        severity_filter=[AlertSeverity.MEDIUM]
    )
    print(f"   ⚠️  Medium alerts: {len(medium_alerts)}")
    
    for alert in medium_alerts:
        print(f"      - {alert.severity.value}: {alert.title}")

def demonstrate_notification_channels(alerting_system):
    """Demonstrate notification channel configuration"""
    
    print("\n📡 Demonstrating notification channels...")
    
    # Show configured channels
    for channel, config in alerting_system.notification_delivery.config.items():
        status = "✅ Enabled" if config.enabled else "❌ Disabled"
        severity_filter = [s.value for s in config.severity_filter]
        print(f"   {channel.value}: {status}")
        print(f"      - Severity filter: {severity_filter}")
    
    # Create a test alert with specific channels
    test_alert = alerting_system.alert_generator.generate_model_drift_alert(
        model_id="test_notification_model",
        accuracy_drop=0.08,
        affected_categories=["test_category"]
    )
    
    # This would normally send notifications, but channels are disabled for demo
    test_alert_id = alerting_system.create_alert(
        test_alert, 
        channels=[NotificationChannel.DASHBOARD]
    )
    print(f"   📨 Test alert created with dashboard notification: {test_alert_id}")

def main():
    """Run the alerting system demo"""
    print("🚨 Starting Autonomous Demand Forecasting Alerting System Demo")
    print("=" * 70)
    
    # Setup demo alerting system
    alerting_system, demo_db_path = setup_demo_alerting_system()
    print("✓ Alerting system initialized with custom configuration")
    
    try:
        # Start services (for escalation and notification delivery)
        alerting_system.start_services()
        print("✓ Alerting services started")
        
        # Generate sample alerts
        alert_ids = generate_sample_alerts(alerting_system)
        
        # Wait a moment for processing
        time.sleep(1)
        
        # Demonstrate alert management
        demonstrate_alert_management(alerting_system, alert_ids)
        
        # Demonstrate severity filtering
        demonstrate_severity_filtering(alerting_system)
        
        # Demonstrate notification channels
        demonstrate_notification_channels(alerting_system)
        
        print("\n📊 Final System Status:")
        print("-" * 30)
        
        # Final statistics
        final_stats = alerting_system.get_alert_statistics()
        print(f"Active alerts: {final_stats['total_active']}")
        print(f"Alert history: {len(alerting_system.alert_history)} total alerts created")
        
        # Show remaining active alerts
        remaining_alerts = alerting_system.get_active_alerts()
        if remaining_alerts:
            print(f"\nRemaining active alerts:")
            for alert in remaining_alerts:
                print(f"  - {alert.severity.value}: {alert.title} ({alert.status.value})")
        
        print("\n✅ Demo completed successfully!")
        print("\nNote: Email, Slack, and Webhook notifications were disabled for this demo.")
        print("In production, these would send real notifications to configured endpoints.")
        
    except KeyboardInterrupt:
        print("\n\n🛑 Demo interrupted by user")
    
    finally:
        # Stop services
        alerting_system.stop_services()
        print("✓ Alerting services stopped")
        
        # Clean up demo database
        if os.path.exists(demo_db_path):
            try:
                os.remove(demo_db_path)
                print("✓ Demo database cleaned up")
            except PermissionError:
                print("⚠️  Demo database cleanup skipped (file in use)")

if __name__ == "__main__":
    main()