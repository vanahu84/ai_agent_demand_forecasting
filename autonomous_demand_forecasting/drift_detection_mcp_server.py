"""
Drift Detection MCP Server for Autonomous Demand Forecasting System.

This server monitors model performance and detects accuracy degradation patterns
to trigger automated retraining workflows.
"""

import asyncio
import json
import logging
import os
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import statistics

import mcp.server.stdio
from dotenv import load_dotenv

# MCP Server Imports
from mcp import types as mcp_types
from mcp.server.lowlevel import NotificationOptions, Server
from mcp.server.models import InitializationOptions

# Import data models
from autonomous_demand_forecasting.database.models import (
    AccuracyMetrics, DriftEvent, SeverityLevel, DriftAnalysis
)

load_dotenv()

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Console output only
    ]
)

# Database path
DATABASE_PATH = os.path.join(os.path.dirname(__file__), "database", "autonomous_demand_forecasting.db")

# Configuration constants
DEFAULT_ACCURACY_THRESHOLD = 0.85
DRIFT_DETECTION_WINDOW_HOURS = 24
PERFORMANCE_HISTORY_DAYS = 30


# --- Database Utility Functions ---
def get_db_connection():
    """Get database connection from the connection pool."""
    try:
        from autonomous_demand_forecasting.database.connection_pool import get_db_connection as get_pooled_connection
        return get_pooled_connection()
    except ImportError:
        # Fallback to direct connection if pool is not available
        conn = sqlite3.connect(DATABASE_PATH)
        conn.row_factory = sqlite3.Row
        return conn

def return_db_connection(conn):
    """Return database connection to the pool."""
    try:
        from autonomous_demand_forecasting.database.connection_pool import return_db_connection as return_pooled_connection
        return_pooled_connection(conn)
    except ImportError:
        # Fallback - just close the connection
        try:
            conn.close()
        except:
            pass


def record_model_performance(
    model_id: str,
    product_category: str,
    accuracy_score: float,
    mape_score: Optional[float] = None,
    rmse_score: Optional[float] = None,
    prediction_count: Optional[int] = None,
    drift_score: Optional[float] = None
) -> Dict[str, Any]:
    """Record model performance metrics in the database."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO model_performance 
            (model_id, product_category, timestamp, accuracy_score, mape_score, 
             rmse_score, prediction_count, drift_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            model_id, product_category, datetime.now(),
            accuracy_score, mape_score, rmse_score, prediction_count, drift_score
        ))
        
        conn.commit()
        performance_id = cursor.lastrowid
        conn.close()
        
        return {
            "success": True,
            "message": f"Performance metrics recorded successfully. ID: {performance_id}",
            "performance_id": performance_id
        }
    except sqlite3.Error as e:
        return {
            "success": False,
            "message": f"Error recording performance metrics: {e}"
        }


def get_model_accuracy_history(
    model_id: str,
    hours_back: int = DRIFT_DETECTION_WINDOW_HOURS
) -> List[AccuracyMetrics]:
    """Retrieve model accuracy history for drift detection."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        cursor.execute("""
            SELECT model_id, product_category, timestamp, accuracy_score, 
                   mape_score, rmse_score, prediction_count
            FROM model_performance 
            WHERE model_id = ? AND timestamp >= ?
            ORDER BY timestamp DESC
        """, (model_id, cutoff_time))
        
        rows = cursor.fetchall()
        conn.close()
        
        metrics_list = []
        for row in rows:
            # Group by category for product_categories list
            categories = [row['product_category']] if row['product_category'] else []
            
            # Handle datetime parsing
            timestamp = row['timestamp']
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp)
            elif not isinstance(timestamp, datetime):
                timestamp = datetime.now()
            
            metrics = AccuracyMetrics(
                model_id=row['model_id'],
                timestamp=timestamp,
                accuracy_score=row['accuracy_score'],
                mape_score=row['mape_score'],
                rmse_score=row['rmse_score'],
                prediction_count=row['prediction_count'],
                product_categories=categories
            )
            metrics_list.append(metrics)
        
        return metrics_list
    except sqlite3.Error as e:
        logging.error(f"Error retrieving accuracy history: {e}")
        return []


def detect_performance_drift(
    model_id: str,
    threshold: float = DEFAULT_ACCURACY_THRESHOLD
) -> List[DriftEvent]:
    """
    Detect performance drift using statistical analysis and configurable thresholds.
    
    Uses multiple detection methods:
    1. Threshold-based detection (accuracy below threshold)
    2. Trend analysis (declining accuracy over time)
    3. Statistical significance testing (sudden drops)
    """
    try:
        # Get recent performance data
        recent_metrics = get_model_accuracy_history(model_id, DRIFT_DETECTION_WINDOW_HOURS)
        
        if not recent_metrics:
            return []
        
        drift_events = []
        
        # Group metrics by category for analysis
        category_metrics = {}
        for metric in recent_metrics:
            for category in metric.product_categories:
                if category not in category_metrics:
                    category_metrics[category] = []
                category_metrics[category].append(metric)
        
        # Analyze each category for drift
        for category, metrics in category_metrics.items():
            if not metrics:
                continue
                
            # Sort metrics by timestamp for trend analysis
            metrics.sort(key=lambda x: x.timestamp)
            recent_accuracies = [m.accuracy_score for m in metrics if m.accuracy_score is not None]
            
            if len(recent_accuracies) < 2:
                continue
            
            # Method 1: Threshold-based detection
            avg_accuracy = statistics.mean(recent_accuracies)
            threshold_drift = avg_accuracy < threshold
            
            # Method 2: Trend analysis - check if accuracy is declining
            trend_drift = False
            if len(recent_accuracies) >= 3:
                # Calculate trend using simple linear regression slope
                x_values = list(range(len(recent_accuracies)))
                n = len(recent_accuracies)
                sum_x = sum(x_values)
                sum_y = sum(recent_accuracies)
                sum_xy = sum(x * y for x, y in zip(x_values, recent_accuracies))
                sum_x2 = sum(x * x for x in x_values)
                
                # Calculate slope
                slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
                trend_drift = slope < -0.01  # Declining trend
            
            # Method 3: Statistical significance - sudden drops
            sudden_drift = False
            if len(recent_accuracies) >= 4:
                # Compare recent half vs earlier half
                mid_point = len(recent_accuracies) // 2
                earlier_half = recent_accuracies[:mid_point]
                recent_half = recent_accuracies[mid_point:]
                
                earlier_avg = statistics.mean(earlier_half)
                recent_avg = statistics.mean(recent_half)
                
                # Check for significant drop (>5% relative decrease)
                if earlier_avg > 0:
                    relative_drop = (earlier_avg - recent_avg) / earlier_avg
                    sudden_drift = relative_drop > 0.05
            
            # Determine if drift occurred
            drift_detected = threshold_drift or trend_drift or sudden_drift
            
            if drift_detected:
                # Calculate accuracy drop
                if threshold_drift:
                    accuracy_drop = threshold - avg_accuracy
                else:
                    # For trend/sudden drift, calculate drop from recent peak
                    max_recent = max(recent_accuracies)
                    accuracy_drop = max_recent - min(recent_accuracies[-3:])  # Last 3 measurements
                
                # Classify severity based on multiple factors
                severity = _classify_drift_severity_advanced(
                    avg_accuracy, accuracy_drop, threshold_drift, trend_drift, sudden_drift
                )
                
                # Calculate comprehensive drift score
                drift_score = _calculate_drift_score(
                    avg_accuracy, accuracy_drop, threshold, len(recent_accuracies)
                )
                
                # Create drift analysis
                drift_analysis = {
                    "detection_methods": {
                        "threshold_based": threshold_drift,
                        "trend_based": trend_drift,
                        "sudden_drop": sudden_drift
                    },
                    "statistics": {
                        "avg_accuracy": avg_accuracy,
                        "min_accuracy": min(recent_accuracies),
                        "max_accuracy": max(recent_accuracies),
                        "accuracy_std": statistics.stdev(recent_accuracies) if len(recent_accuracies) > 1 else 0,
                        "sample_count": len(recent_accuracies)
                    },
                    "trend_analysis": {
                        "slope": slope if 'slope' in locals() else None,
                        "declining_trend": trend_drift
                    }
                }
                
                drift_event = DriftEvent(
                    model_id=model_id,
                    severity=severity,
                    detected_at=datetime.now(),
                    accuracy_drop=accuracy_drop,
                    affected_categories=[category],
                    drift_score=drift_score,
                    drift_analysis=drift_analysis
                )
                
                drift_events.append(drift_event)
        
        return drift_events
    except Exception as e:
        logging.error(f"Error detecting performance drift: {e}")
        return []


def _classify_drift_severity_advanced(
    avg_accuracy: float,
    accuracy_drop: float, 
    threshold_drift: bool,
    trend_drift: bool,
    sudden_drift: bool
) -> SeverityLevel:
    """Advanced drift severity classification considering multiple factors."""
    
    # High severity conditions
    if (avg_accuracy < 0.70 or  # Very low accuracy
        accuracy_drop > 0.15):  # Large drop
        return SeverityLevel.HIGH
    
    # Medium severity conditions  
    elif (avg_accuracy < 0.75 or  # Moderately low accuracy
          accuracy_drop > 0.08 or  # Moderate drop
          (threshold_drift and trend_drift)):  # Threshold + trend
        return SeverityLevel.MEDIUM
    
    # Low severity (any drift detected but not severe)
    else:
        return SeverityLevel.LOW


def _calculate_drift_score(
    avg_accuracy: float,
    accuracy_drop: float,
    threshold: float,
    sample_count: int
) -> float:
    """Calculate comprehensive drift score (0-1, higher = more severe drift)."""
    
    # Base score from accuracy drop relative to threshold
    base_score = min(accuracy_drop / (threshold * 0.5), 1.0)  # Scale relative to half threshold
    
    # Adjust for absolute accuracy level (how far below threshold)
    accuracy_factor = max(0, (threshold - avg_accuracy) / threshold)
    
    # Adjust for sample size (more samples = more confidence)
    confidence_factor = min(sample_count / 5.0, 1.0)  # Max confidence at 5+ samples
    
    # Combine factors with higher weight on accuracy drop
    drift_score = (base_score * 0.6 + accuracy_factor * 0.3 + confidence_factor * 0.1)
    
    return min(drift_score, 1.0)


def detect_real_time_drift(
    model_id: str,
    current_accuracy: float,
    product_category: str,
    threshold: float = DEFAULT_ACCURACY_THRESHOLD
) -> Optional[DriftEvent]:
    """
    Real-time drift detection for immediate alerting.
    
    Analyzes current accuracy against recent history for immediate drift detection.
    """
    try:
        # Get recent history for comparison
        recent_metrics = get_model_accuracy_history(model_id, hours_back=6)  # Last 6 hours
        
        if not recent_metrics:
            # No history - check against threshold only
            if current_accuracy < threshold:
                accuracy_drop = threshold - current_accuracy
                severity = classify_drift_severity(accuracy_drop)
                
                return DriftEvent(
                    model_id=model_id,
                    severity=severity,
                    detected_at=datetime.now(),
                    accuracy_drop=accuracy_drop,
                    affected_categories=[product_category],
                    drift_score=accuracy_drop / threshold
                )
            return None
        
        # Filter metrics for the same category
        category_metrics = [m for m in recent_metrics 
                          if product_category in m.product_categories and m.accuracy_score is not None]
        
        if not category_metrics:
            # No category history - check against threshold
            if current_accuracy < threshold:
                accuracy_drop = threshold - current_accuracy
                severity = classify_drift_severity(accuracy_drop)
                
                return DriftEvent(
                    model_id=model_id,
                    severity=severity,
                    detected_at=datetime.now(),
                    accuracy_drop=accuracy_drop,
                    affected_categories=[product_category],
                    drift_score=accuracy_drop / threshold
                )
            return None
        
        # Calculate recent average for comparison
        recent_accuracies = [m.accuracy_score for m in category_metrics]
        recent_avg = statistics.mean(recent_accuracies)
        
        # Check for significant drop from recent performance
        performance_drop = recent_avg - current_accuracy
        threshold_drop = threshold - current_accuracy
        
        # Use the larger drop for severity assessment
        accuracy_drop = max(performance_drop, threshold_drop) if threshold_drop > 0 else performance_drop
        
        # Detect drift if current accuracy is significantly below recent average OR below threshold
        drift_detected = (performance_drop > 0.03 or  # 3% drop from recent average
                         current_accuracy < threshold)
        
        if drift_detected and accuracy_drop > 0:
            severity = classify_drift_severity(accuracy_drop)
            drift_score = _calculate_drift_score(current_accuracy, accuracy_drop, threshold, len(recent_accuracies))
            
            return DriftEvent(
                model_id=model_id,
                severity=severity,
                detected_at=datetime.now(),
                accuracy_drop=accuracy_drop,
                affected_categories=[product_category],
                drift_score=drift_score,
                drift_analysis={
                    "real_time_detection": True,
                    "current_accuracy": current_accuracy,
                    "recent_average": recent_avg,
                    "performance_drop": performance_drop,
                    "threshold_violation": current_accuracy < threshold
                }
            )
        
        return None
        
    except Exception as e:
        logging.error(f"Error in real-time drift detection: {e}")
        return None


def record_drift_event(drift_event: DriftEvent) -> Dict[str, Any]:
    """Record a drift event in the database."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO drift_events 
            (model_id, severity, detected_at, accuracy_drop, affected_categories, drift_analysis)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            drift_event.model_id,
            drift_event.severity.value,
            drift_event.detected_at,
            drift_event.accuracy_drop,
            json.dumps(drift_event.affected_categories),
            json.dumps(drift_event.drift_analysis) if drift_event.drift_analysis else None
        ))
        
        conn.commit()
        event_id = cursor.lastrowid
        conn.close()
        
        return {
            "success": True,
            "message": f"Drift event recorded successfully. ID: {event_id}",
            "event_id": event_id
        }
    except sqlite3.Error as e:
        return {
            "success": False,
            "message": f"Error recording drift event: {e}"
        }


def get_drift_events_history(
    model_id: str,
    days_back: int = PERFORMANCE_HISTORY_DAYS
) -> List[DriftEvent]:
    """Retrieve historical drift events for analysis."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cutoff_time = datetime.now() - timedelta(days=days_back)
        
        cursor.execute("""
            SELECT model_id, severity, detected_at, resolved_at, accuracy_drop, 
                   affected_categories, drift_analysis
            FROM drift_events 
            WHERE model_id = ? AND detected_at >= ?
            ORDER BY detected_at DESC
        """, (model_id, cutoff_time))
        
        rows = cursor.fetchall()
        conn.close()
        
        events = []
        for row in rows:
            affected_categories = json.loads(row['affected_categories']) if row['affected_categories'] else []
            drift_analysis = json.loads(row['drift_analysis']) if row['drift_analysis'] else None
            
            # Handle datetime parsing
            detected_at = row['detected_at']
            if isinstance(detected_at, str):
                detected_at = datetime.fromisoformat(detected_at)
            elif not isinstance(detected_at, datetime):
                detected_at = datetime.now()
                
            resolved_at = None
            if row['resolved_at']:
                if isinstance(row['resolved_at'], str):
                    resolved_at = datetime.fromisoformat(row['resolved_at'])
                elif isinstance(row['resolved_at'], datetime):
                    resolved_at = row['resolved_at']
            
            event = DriftEvent(
                model_id=row['model_id'],
                severity=SeverityLevel(row['severity']),
                detected_at=detected_at,
                resolved_at=resolved_at,
                accuracy_drop=row['accuracy_drop'],
                affected_categories=affected_categories,
                drift_analysis=drift_analysis
            )
            events.append(event)
        
        return events
    except sqlite3.Error as e:
        logging.error(f"Error retrieving drift events history: {e}")
        return []


def analyze_drift_patterns(model_id: str) -> DriftAnalysis:
    """Analyze drift patterns and provide recommendations."""
    try:
        # Get recent drift events
        drift_events = get_drift_events_history(model_id, PERFORMANCE_HISTORY_DAYS)
        
        # Get accuracy trends
        accuracy_history = get_model_accuracy_history(model_id, DRIFT_DETECTION_WINDOW_HOURS * 7)  # 7 days
        
        # Build accuracy trends by category
        accuracy_trends = {}
        for metric in accuracy_history:
            for category in metric.product_categories:
                if category not in accuracy_trends:
                    accuracy_trends[category] = []
                if metric.accuracy_score is not None:
                    accuracy_trends[category].append(metric.accuracy_score)
        
        # Identify affected categories
        affected_categories = list(set(
            cat for event in drift_events for cat in event.affected_categories
        ))
        
        # Generate recommendations
        recommendations = []
        if drift_events:
            high_severity_events = [e for e in drift_events if e.severity == SeverityLevel.HIGH]
            medium_severity_events = [e for e in drift_events if e.severity == SeverityLevel.MEDIUM]
            
            if high_severity_events:
                recommendations.append("Immediate model retraining required due to HIGH severity drift")
                recommendations.append("Consider expanding training dataset with recent data")
            elif medium_severity_events:
                recommendations.append("Schedule model retraining within 24 hours")
                recommendations.append("Monitor performance closely for further degradation")
            else:
                recommendations.append("Continue monitoring - LOW severity drift detected")
                recommendations.append("Consider data quality checks and feature engineering")
        
        analysis = DriftAnalysis(
            model_id=model_id,
            analysis_timestamp=datetime.now(),
            drift_events=drift_events,
            accuracy_trends=accuracy_trends,
            affected_categories=affected_categories,
            recommended_actions=recommendations
        )
        
        return analysis
    except Exception as e:
        logging.error(f"Error analyzing drift patterns: {e}")
        return DriftAnalysis(
            model_id=model_id,
            analysis_timestamp=datetime.now(),
            recommended_actions=[f"Error in analysis: {str(e)}"]
        )


def analyze_historical_performance_trends(
    model_id: str,
    days_back: int = PERFORMANCE_HISTORY_DAYS
) -> Dict[str, Any]:
    """
    Analyze historical performance trends for comprehensive insights.
    
    Provides detailed trend analysis including:
    - Performance degradation patterns
    - Seasonal variations
    - Category-specific trends
    - Prediction accuracy patterns
    """
    try:
        # Get extended history for trend analysis
        accuracy_history = get_model_accuracy_history(model_id, days_back * 24)
        
        if not accuracy_history:
            return {
                "success": False,
                "message": "No historical data available for trend analysis"
            }
        
        # Group by category and time periods
        category_trends = {}
        daily_trends = {}
        
        for metric in accuracy_history:
            # Category-based analysis
            for category in metric.product_categories:
                if category not in category_trends:
                    category_trends[category] = {
                        "accuracies": [],
                        "timestamps": [],
                        "mape_scores": [],
                        "rmse_scores": []
                    }
                
                category_trends[category]["accuracies"].append(metric.accuracy_score or 0)
                category_trends[category]["timestamps"].append(metric.timestamp)
                if metric.mape_score:
                    category_trends[category]["mape_scores"].append(metric.mape_score)
                if metric.rmse_score:
                    category_trends[category]["rmse_scores"].append(metric.rmse_score)
            
            # Daily aggregation
            day_key = metric.timestamp.strftime("%Y-%m-%d")
            if day_key not in daily_trends:
                daily_trends[day_key] = []
            if metric.accuracy_score:
                daily_trends[day_key].append(metric.accuracy_score)
        
        # Calculate trend statistics
        trend_analysis = {}
        
        for category, data in category_trends.items():
            if len(data["accuracies"]) < 2:
                continue
                
            accuracies = data["accuracies"]
            
            # Calculate trend metrics
            avg_accuracy = statistics.mean(accuracies)
            min_accuracy = min(accuracies)
            max_accuracy = max(accuracies)
            accuracy_std = statistics.stdev(accuracies) if len(accuracies) > 1 else 0
            
            # Calculate trend slope
            x_values = list(range(len(accuracies)))
            n = len(accuracies)
            if n > 1:
                sum_x = sum(x_values)
                sum_y = sum(accuracies)
                sum_xy = sum(x * y for x, y in zip(x_values, accuracies))
                sum_x2 = sum(x * x for x in x_values)
                
                slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
            else:
                slope = 0
            
            # Identify trend direction
            if slope > 0.005:
                trend_direction = "improving"
            elif slope < -0.005:
                trend_direction = "declining"
            else:
                trend_direction = "stable"
            
            # Calculate volatility (coefficient of variation)
            volatility = (accuracy_std / avg_accuracy) if avg_accuracy > 0 else 0
            
            trend_analysis[category] = {
                "avg_accuracy": avg_accuracy,
                "min_accuracy": min_accuracy,
                "max_accuracy": max_accuracy,
                "accuracy_range": max_accuracy - min_accuracy,
                "accuracy_std": accuracy_std,
                "trend_slope": slope,
                "trend_direction": trend_direction,
                "volatility": volatility,
                "sample_count": len(accuracies),
                "avg_mape": statistics.mean(data["mape_scores"]) if data["mape_scores"] else None,
                "avg_rmse": statistics.mean(data["rmse_scores"]) if data["rmse_scores"] else None
            }
        
        # Daily trend analysis
        daily_analysis = {}
        for day, accuracies in daily_trends.items():
            if accuracies:
                daily_analysis[day] = {
                    "avg_accuracy": statistics.mean(accuracies),
                    "min_accuracy": min(accuracies),
                    "max_accuracy": max(accuracies),
                    "prediction_count": len(accuracies)
                }
        
        # Generate insights
        insights = []
        
        # Overall model health
        all_accuracies = [acc for data in category_trends.values() for acc in data["accuracies"]]
        if all_accuracies:
            overall_avg = statistics.mean(all_accuracies)
            if overall_avg > 0.90:
                insights.append("Model shows excellent overall performance")
            elif overall_avg > 0.80:
                insights.append("Model shows good overall performance")
            elif overall_avg > 0.70:
                insights.append("Model shows moderate performance - consider optimization")
            else:
                insights.append("Model shows poor performance - immediate attention required")
        
        # Category-specific insights
        declining_categories = [cat for cat, data in trend_analysis.items() 
                              if data["trend_direction"] == "declining"]
        if declining_categories:
            insights.append(f"Declining performance in categories: {', '.join(declining_categories)}")
        
        improving_categories = [cat for cat, data in trend_analysis.items() 
                              if data["trend_direction"] == "improving"]
        if improving_categories:
            insights.append(f"Improving performance in categories: {', '.join(improving_categories)}")
        
        # Volatility insights
        high_volatility_categories = [cat for cat, data in trend_analysis.items() 
                                    if data["volatility"] > 0.1]
        if high_volatility_categories:
            insights.append(f"High volatility in categories: {', '.join(high_volatility_categories)}")
        
        return {
            "success": True,
            "model_id": model_id,
            "analysis_period_days": days_back,
            "category_trends": trend_analysis,
            "daily_trends": daily_analysis,
            "insights": insights,
            "total_data_points": len(accuracy_history),
            "analyzed_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logging.error(f"Error in historical performance trend analysis: {e}")
        return {
            "success": False,
            "message": f"Error in trend analysis: {str(e)}"
        }


def generate_performance_alert(
    model_id: str,
    drift_event: DriftEvent,
    alert_level: str = "WARNING"
) -> Dict[str, Any]:
    """
    Generate structured performance alert for escalation.
    
    Creates detailed alerts for different stakeholders based on drift severity.
    """
    try:
        # Determine alert recipients and urgency based on severity
        if drift_event.severity == SeverityLevel.HIGH:
            alert_level = "CRITICAL"
            recipients = ["data_science_team", "operations_team", "management"]
            urgency = "immediate"
            escalation_time = 15  # minutes
        elif drift_event.severity == SeverityLevel.MEDIUM:
            alert_level = "WARNING"
            recipients = ["data_science_team", "operations_team"]
            urgency = "high"
            escalation_time = 60  # minutes
        else:
            alert_level = "INFO"
            recipients = ["data_science_team"]
            urgency = "normal"
            escalation_time = 240  # minutes
        
        # Generate alert message
        alert_message = f"""
Model Performance Alert - {alert_level}

Model ID: {model_id}
Severity: {drift_event.severity.value}
Detected At: {drift_event.detected_at.strftime('%Y-%m-%d %H:%M:%S')}
Accuracy Drop: {drift_event.accuracy_drop:.3f} ({drift_event.accuracy_drop*100:.1f}%)
Affected Categories: {', '.join(drift_event.affected_categories)}
Drift Score: {drift_event.drift_score:.3f}

Detection Methods:
"""
        
        if drift_event.drift_analysis:
            detection_methods = drift_event.drift_analysis.get("detection_methods", {})
            for method, detected in detection_methods.items():
                status = "✓" if detected else "✗"
                alert_message += f"  {status} {method.replace('_', ' ').title()}\n"
            
            stats = drift_event.drift_analysis.get("statistics", {})
            if stats:
                alert_message += f"\nPerformance Statistics:\n"
                alert_message += f"  Average Accuracy: {stats.get('avg_accuracy', 0):.3f}\n"
                alert_message += f"  Min Accuracy: {stats.get('min_accuracy', 0):.3f}\n"
                alert_message += f"  Max Accuracy: {stats.get('max_accuracy', 0):.3f}\n"
                alert_message += f"  Sample Count: {stats.get('sample_count', 0)}\n"
        
        # Add recommendations
        analysis = analyze_drift_patterns(model_id)
        if analysis.recommended_actions:
            alert_message += f"\nRecommended Actions:\n"
            for i, action in enumerate(analysis.recommended_actions, 1):
                alert_message += f"  {i}. {action}\n"
        
        alert = {
            "alert_id": f"drift_{model_id}_{int(drift_event.detected_at.timestamp())}",
            "model_id": model_id,
            "alert_level": alert_level,
            "severity": drift_event.severity.value,
            "urgency": urgency,
            "recipients": recipients,
            "escalation_time_minutes": escalation_time,
            "created_at": datetime.now().isoformat(),
            "drift_event_id": drift_event.id,
            "message": alert_message.strip(),
            "metadata": {
                "accuracy_drop": drift_event.accuracy_drop,
                "drift_score": drift_event.drift_score,
                "affected_categories": drift_event.affected_categories,
                "detection_timestamp": drift_event.detected_at.isoformat()
            }
        }
        
        return alert
        
    except Exception as e:
        logging.error(f"Error generating performance alert: {e}")
        return {
            "alert_id": f"error_{model_id}_{int(datetime.now().timestamp())}",
            "alert_level": "ERROR",
            "message": f"Failed to generate alert for model {model_id}: {str(e)}",
            "created_at": datetime.now().isoformat()
        }


def identify_root_causes(
    model_id: str,
    drift_event: DriftEvent
) -> Dict[str, Any]:
    """
    Identify potential root causes of model drift.
    
    Analyzes patterns and provides insights into why drift occurred.
    """
    try:
        root_causes = []
        confidence_scores = {}
        
        # Get recent performance history for analysis
        recent_history = get_model_accuracy_history(model_id, DRIFT_DETECTION_WINDOW_HOURS * 2)
        
        if not recent_history:
            return {
                "success": False,
                "message": "Insufficient data for root cause analysis"
            }
        
        # Analyze drift patterns from drift_analysis
        if drift_event.drift_analysis:
            detection_methods = drift_event.drift_analysis.get("detection_methods", {})
            stats = drift_event.drift_analysis.get("statistics", {})
            
            # Pattern 1: Sudden drop suggests data quality issues or external changes
            if detection_methods.get("sudden_drop", False):
                root_causes.append({
                    "cause": "Data Quality Issues",
                    "description": "Sudden performance drop suggests potential data quality problems or external system changes",
                    "indicators": ["sudden_drop_detected", "performance_cliff"],
                    "recommendations": [
                        "Check data pipeline for anomalies",
                        "Validate recent data sources",
                        "Review external system changes"
                    ]
                })
                confidence_scores["data_quality"] = 0.8
            
            # Pattern 2: Gradual decline suggests concept drift
            if detection_methods.get("trend_based", False) and not detection_methods.get("sudden_drop", False):
                root_causes.append({
                    "cause": "Concept Drift",
                    "description": "Gradual performance decline indicates changing patterns in the underlying data",
                    "indicators": ["declining_trend", "gradual_degradation"],
                    "recommendations": [
                        "Analyze recent market trends",
                        "Update feature engineering",
                        "Retrain with recent data"
                    ]
                })
                confidence_scores["concept_drift"] = 0.7
            
            # Pattern 3: High volatility suggests model instability
            if stats.get("accuracy_std", 0) > 0.05:  # High standard deviation
                root_causes.append({
                    "cause": "Model Instability",
                    "description": "High performance volatility indicates model sensitivity to input variations",
                    "indicators": ["high_variance", "inconsistent_predictions"],
                    "recommendations": [
                        "Review model regularization",
                        "Increase training data diversity",
                        "Consider ensemble methods"
                    ]
                })
                confidence_scores["model_instability"] = 0.6
            
            # Pattern 4: Category-specific drift suggests domain-specific issues
            if len(drift_event.affected_categories) == 1:
                category = drift_event.affected_categories[0]
                root_causes.append({
                    "cause": "Category-Specific Issues",
                    "description": f"Drift isolated to {category} category suggests domain-specific problems",
                    "indicators": ["single_category_impact", f"category_{category}_affected"],
                    "recommendations": [
                        f"Analyze {category} category data quality",
                        f"Review {category} business rules",
                        f"Check {category} market conditions"
                    ]
                })
                confidence_scores["category_specific"] = 0.75
        
        # Analyze temporal patterns
        if len(recent_history) >= 5:
            # Check for time-based patterns
            timestamps = [m.timestamp for m in recent_history]
            accuracies = [m.accuracy_score for m in recent_history if m.accuracy_score is not None]
            
            if len(accuracies) >= 5:
                # Check for cyclical patterns (daily/weekly)
                hours = [t.hour for t in timestamps]
                weekdays = [t.weekday() for t in timestamps]
                
                # Simple pattern detection
                if len(set(hours)) < len(hours) * 0.5:  # Concentrated in specific hours
                    root_causes.append({
                        "cause": "Temporal Patterns",
                        "description": "Performance issues concentrated in specific time periods",
                        "indicators": ["time_based_degradation", "cyclical_patterns"],
                        "recommendations": [
                            "Analyze time-based data patterns",
                            "Consider temporal features",
                            "Review system load patterns"
                        ]
                    })
                    confidence_scores["temporal_patterns"] = 0.5
        
        # If no specific patterns identified, provide general causes
        if not root_causes:
            root_causes.append({
                "cause": "General Performance Degradation",
                "description": "Model performance has declined but specific patterns are unclear",
                "indicators": ["threshold_violation", "general_drift"],
                "recommendations": [
                    "Perform comprehensive data analysis",
                    "Review model assumptions",
                    "Consider model retraining"
                ]
            })
            confidence_scores["general"] = 0.4
        
        # Rank causes by confidence
        ranked_causes = sorted(root_causes, 
                             key=lambda x: confidence_scores.get(x["cause"].lower().replace(" ", "_"), 0), 
                             reverse=True)
        
        return {
            "success": True,
            "model_id": model_id,
            "drift_event_id": drift_event.id,
            "analysis_timestamp": datetime.now().isoformat(),
            "root_causes": ranked_causes,
            "confidence_scores": confidence_scores,
            "primary_cause": ranked_causes[0] if ranked_causes else None,
            "analysis_summary": f"Identified {len(root_causes)} potential root causes for model drift"
        }
        
    except Exception as e:
        logging.error(f"Error in root cause analysis: {e}")
        return {
            "success": False,
            "message": f"Error in root cause analysis: {str(e)}"
        }


def classify_drift_severity(accuracy_drop: float) -> SeverityLevel:
    """Classify drift severity based on accuracy drop."""
    if accuracy_drop >= 0.15:  # 15% or more drop
        return SeverityLevel.HIGH
    elif accuracy_drop >= 0.05:  # 5-15% drop
        return SeverityLevel.MEDIUM
    else:  # Less than 5% drop
        return SeverityLevel.LOW


# --- MCP Server Setup ---
logging.info("Creating Drift Detection MCP Server instance...")
app = Server("drift-detection-mcp-server")


@app.list_tools()
async def list_mcp_tools() -> list[mcp_types.Tool]:
    """MCP handler to list tools this server exposes."""
    logging.info("Drift Detection MCP Server: Received list_tools request.")
    
    tools = [
        mcp_types.Tool(
            name="monitor_model_accuracy",
            description="Monitor and record model accuracy metrics for drift detection",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": {
                        "type": "string",
                        "description": "The ID of the model to monitor"
                    },
                    "product_category": {
                        "type": "string",
                        "description": "The product category for this accuracy measurement"
                    },
                    "accuracy_score": {
                        "type": "number",
                        "description": "The accuracy score (0.0 to 1.0)"
                    },
                    "mape_score": {
                        "type": "number",
                        "description": "Mean Absolute Percentage Error score (optional)"
                    },
                    "rmse_score": {
                        "type": "number",
                        "description": "Root Mean Square Error score (optional)"
                    },
                    "prediction_count": {
                        "type": "integer",
                        "description": "Number of predictions made (optional)"
                    }
                },
                "required": ["model_id", "product_category", "accuracy_score"]
            }
        ),
        mcp_types.Tool(
            name="detect_performance_drift",
            description="Detect performance drift based on accuracy thresholds",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": {
                        "type": "string",
                        "description": "The ID of the model to check for drift"
                    },
                    "threshold": {
                        "type": "number",
                        "description": "Accuracy threshold below which drift is detected (default: 0.85)",
                        "default": DEFAULT_ACCURACY_THRESHOLD
                    }
                },
                "required": ["model_id"]
            }
        ),
        mcp_types.Tool(
            name="analyze_drift_patterns",
            description="Analyze drift patterns and provide recommendations",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": {
                        "type": "string",
                        "description": "The ID of the model to analyze"
                    }
                },
                "required": ["model_id"]
            }
        ),
        mcp_types.Tool(
            name="classify_drift_severity",
            description="Classify drift severity based on accuracy drop",
            inputSchema={
                "type": "object",
                "properties": {
                    "accuracy_drop": {
                        "type": "number",
                        "description": "The accuracy drop amount (e.g., 0.1 for 10% drop)"
                    }
                },
                "required": ["accuracy_drop"]
            }
        ),
        mcp_types.Tool(
            name="get_model_accuracy_history",
            description="Retrieve model accuracy history for analysis",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": {
                        "type": "string",
                        "description": "The ID of the model"
                    },
                    "hours_back": {
                        "type": "integer",
                        "description": "Number of hours back to retrieve history (default: 24)",
                        "default": DRIFT_DETECTION_WINDOW_HOURS
                    }
                },
                "required": ["model_id"]
            }
        ),
        mcp_types.Tool(
            name="detect_real_time_drift",
            description="Real-time drift detection for immediate alerting",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": {
                        "type": "string",
                        "description": "The ID of the model"
                    },
                    "current_accuracy": {
                        "type": "number",
                        "description": "Current accuracy score to check for drift"
                    },
                    "product_category": {
                        "type": "string",
                        "description": "Product category for this accuracy measurement"
                    },
                    "threshold": {
                        "type": "number",
                        "description": "Accuracy threshold for drift detection (default: 0.85)",
                        "default": DEFAULT_ACCURACY_THRESHOLD
                    }
                },
                "required": ["model_id", "current_accuracy", "product_category"]
            }
        ),
        mcp_types.Tool(
            name="analyze_historical_performance_trends",
            description="Analyze historical performance trends for comprehensive insights",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": {
                        "type": "string",
                        "description": "The ID of the model to analyze"
                    },
                    "days_back": {
                        "type": "integer",
                        "description": "Number of days back to analyze (default: 30)",
                        "default": PERFORMANCE_HISTORY_DAYS
                    }
                },
                "required": ["model_id"]
            }
        ),
        mcp_types.Tool(
            name="generate_performance_alert",
            description="Generate structured performance alert for escalation",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": {
                        "type": "string",
                        "description": "The ID of the model"
                    },
                    "drift_event": {
                        "type": "object",
                        "description": "Drift event data (JSON object with drift event details)"
                    },
                    "alert_level": {
                        "type": "string",
                        "description": "Alert level override (default: auto-determined)",
                        "default": "WARNING"
                    }
                },
                "required": ["model_id", "drift_event"]
            }
        ),
        mcp_types.Tool(
            name="identify_root_causes",
            description="Identify potential root causes of model drift",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": {
                        "type": "string",
                        "description": "The ID of the model"
                    },
                    "drift_event": {
                        "type": "object",
                        "description": "Drift event data (JSON object with drift event details)"
                    }
                },
                "required": ["model_id", "drift_event"]
            }
        )
    ]
    
    return tools


@app.call_tool()
async def call_mcp_tool(name: str, arguments: dict) -> list[mcp_types.TextContent]:
    """MCP handler to execute a tool call requested by an MCP client."""
    logging.info(f"Drift Detection MCP Server: Received call_tool request for '{name}' with args: {arguments}")

    try:
        if name == "monitor_model_accuracy":
            result = record_model_performance(
                model_id=arguments.get("model_id"),
                product_category=arguments.get("product_category"),
                accuracy_score=arguments.get("accuracy_score"),
                mape_score=arguments.get("mape_score"),
                rmse_score=arguments.get("rmse_score"),
                prediction_count=arguments.get("prediction_count")
            )
        
        elif name == "detect_performance_drift":
            drift_events = detect_performance_drift(
                model_id=arguments.get("model_id"),
                threshold=arguments.get("threshold", DEFAULT_ACCURACY_THRESHOLD)
            )
            
            # Record detected drift events
            recorded_events = []
            for event in drift_events:
                record_result = record_drift_event(event)
                if record_result["success"]:
                    recorded_events.append(event.to_dict())
            
            result = {
                "success": True,
                "message": f"Drift detection completed. Found {len(drift_events)} drift events.",
                "drift_events": recorded_events,
                "drift_detected": len(drift_events) > 0
            }
        
        elif name == "analyze_drift_patterns":
            analysis = analyze_drift_patterns(arguments.get("model_id"))
            result = {
                "success": True,
                "message": "Drift pattern analysis completed.",
                "analysis": {
                    "model_id": analysis.model_id,
                    "analysis_timestamp": analysis.analysis_timestamp.isoformat(),
                    "drift_events_count": len(analysis.drift_events),
                    "highest_severity": analysis.highest_severity.value if analysis.highest_severity else None,
                    "affected_categories": analysis.affected_categories,
                    "accuracy_trends": analysis.accuracy_trends,
                    "recommended_actions": analysis.recommended_actions
                }
            }
        
        elif name == "classify_drift_severity":
            severity = classify_drift_severity(arguments.get("accuracy_drop"))
            result = {
                "success": True,
                "message": f"Drift severity classified as {severity.value}",
                "severity": severity.value,
                "accuracy_drop": arguments.get("accuracy_drop")
            }
        
        elif name == "get_model_accuracy_history":
            metrics_history = get_model_accuracy_history(
                model_id=arguments.get("model_id"),
                hours_back=arguments.get("hours_back", DRIFT_DETECTION_WINDOW_HOURS)
            )
            
            result = {
                "success": True,
                "message": f"Retrieved {len(metrics_history)} accuracy records.",
                "metrics": [metric.to_dict() for metric in metrics_history]
            }
        
        elif name == "detect_real_time_drift":
            drift_event = detect_real_time_drift(
                model_id=arguments.get("model_id"),
                current_accuracy=arguments.get("current_accuracy"),
                product_category=arguments.get("product_category"),
                threshold=arguments.get("threshold", DEFAULT_ACCURACY_THRESHOLD)
            )
            
            if drift_event:
                # Record the drift event
                record_result = record_drift_event(drift_event)
                result = {
                    "success": True,
                    "message": "Real-time drift detected and recorded.",
                    "drift_detected": True,
                    "drift_event": drift_event.to_dict(),
                    "record_result": record_result
                }
            else:
                result = {
                    "success": True,
                    "message": "No drift detected in real-time analysis.",
                    "drift_detected": False
                }
        
        elif name == "analyze_historical_performance_trends":
            result = analyze_historical_performance_trends(
                model_id=arguments.get("model_id"),
                days_back=arguments.get("days_back", PERFORMANCE_HISTORY_DAYS)
            )
        
        elif name == "generate_performance_alert":
            # Convert drift_event dict back to DriftEvent object
            drift_event_data = arguments.get("drift_event")
            try:
                drift_event = DriftEvent(
                    model_id=drift_event_data["model_id"],
                    severity=SeverityLevel(drift_event_data["severity"]),
                    detected_at=datetime.fromisoformat(drift_event_data["detected_at"]),
                    accuracy_drop=drift_event_data["accuracy_drop"],
                    affected_categories=drift_event_data.get("affected_categories", []),
                    drift_score=drift_event_data.get("drift_score", 0),
                    drift_analysis=drift_event_data.get("drift_analysis")
                )
                
                alert = generate_performance_alert(
                    model_id=arguments.get("model_id"),
                    drift_event=drift_event,
                    alert_level=arguments.get("alert_level", "WARNING")
                )
                
                result = {
                    "success": True,
                    "message": "Performance alert generated successfully.",
                    "alert": alert
                }
            except Exception as e:
                result = {
                    "success": False,
                    "message": f"Error generating alert: {str(e)}"
                }
        
        elif name == "identify_root_causes":
            # Convert drift_event dict back to DriftEvent object
            drift_event_data = arguments.get("drift_event")
            try:
                drift_event = DriftEvent(
                    model_id=drift_event_data["model_id"],
                    severity=SeverityLevel(drift_event_data["severity"]),
                    detected_at=datetime.fromisoformat(drift_event_data["detected_at"]),
                    accuracy_drop=drift_event_data["accuracy_drop"],
                    affected_categories=drift_event_data.get("affected_categories", []),
                    drift_score=drift_event_data.get("drift_score", 0),
                    drift_analysis=drift_event_data.get("drift_analysis")
                )
                
                result = identify_root_causes(
                    model_id=arguments.get("model_id"),
                    drift_event=drift_event
                )
            except Exception as e:
                result = {
                    "success": False,
                    "message": f"Error in root cause analysis: {str(e)}"
                }
            
        
        else:
            result = {
                "success": False,
                "message": f"Tool '{name}' not implemented by this server.",
                "available_tools": [
                    "monitor_model_accuracy", "detect_performance_drift", 
                    "analyze_drift_patterns", "classify_drift_severity", 
                    "get_model_accuracy_history", "detect_real_time_drift",
                    "analyze_historical_performance_trends", "generate_performance_alert",
                    "identify_root_causes"
                ]
            }
        
        logging.info(f"Drift Detection MCP Server: Tool '{name}' executed successfully")
        response_text = json.dumps(result, indent=2)
        return [mcp_types.TextContent(type="text", text=response_text)]

    except Exception as e:
        logging.error(f"Drift Detection MCP Server: Error executing tool '{name}': {e}", exc_info=True)
        error_payload = {
            "success": False,
            "message": f"Failed to execute tool '{name}': {str(e)}",
            "tool_name": name
        }
        error_text = json.dumps(error_payload)
        return [mcp_types.TextContent(type="text", text=error_text)]


# --- MCP Server Runner ---
async def run_mcp_stdio_server():
    """Runs the MCP server, listening for connections over standard input/output."""
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        logging.info("Drift Detection MCP Stdio Server: Starting handshake with client...")
        await app.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name=app.name,
                server_version="0.1.0",
                capabilities=app.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )
        logging.info("Drift Detection MCP Stdio Server: Run loop finished or client disconnected.")


if __name__ == "__main__":
    logging.info("Launching Drift Detection MCP Server via stdio...")
    try:
        asyncio.run(run_mcp_stdio_server())
    except KeyboardInterrupt:
        logging.info("\nDrift Detection MCP Server (stdio) stopped by user.")
    except Exception as e:
        logging.critical(f"Drift Detection MCP Server (stdio) encountered an unhandled error: {e}", exc_info=True)
    finally:
        logging.info("Drift Detection MCP Server (stdio) process exiting.")