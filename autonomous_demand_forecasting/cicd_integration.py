"""
CI/CD Integration Module for MLOps Pipeline.

This module provides integration with continuous integration and deployment systems,
implementing automated pipeline triggers, quality gates, and deployment orchestration.
"""

import asyncio
import json
import logging
import os
import yaml
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

from autonomous_demand_forecasting.mlops_pipeline import MLOpsPipeline, PipelineStatus
from autonomous_demand_forecasting.git_integration import GitIntegration
from autonomous_demand_forecasting.adaptive_optimization import adaptive_optimizer, LearningFeedback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TriggerType(Enum):
    """Pipeline trigger types."""
    DRIFT_DETECTION = "drift_detection"
    SCHEDULED = "scheduled"
    MANUAL = "manual"
    DATA_QUALITY = "data_quality"
    GIT_WEBHOOK = "git_webhook"


class QualityGateStatus(Enum):
    """Quality gate status."""
    PASSED = "PASSED"
    FAILED = "FAILED"
    PENDING = "PENDING"


@dataclass
class QualityGateResult:
    """Quality gate evaluation result."""
    gate_name: str
    status: QualityGateStatus
    score: float
    threshold: float
    message: str
    details: Dict[str, Any] = None


@dataclass
class PipelineTrigger:
    """Pipeline trigger configuration."""
    trigger_type: TriggerType
    enabled: bool
    conditions: Dict[str, Any]
    cooldown_hours: Optional[int] = None
    last_triggered: Optional[datetime] = None


class CICDIntegration:
    """CI/CD integration system for MLOps pipeline."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize CI/CD integration."""
        self.config_path = config_path or os.path.join(
            os.path.dirname(__file__), "config", "cicd_config.yaml"
        )
        self.config = self._load_config()
        self.mlops_pipeline = MLOpsPipeline()
        self.git_integration = GitIntegration()
        self.adaptive_optimizer = adaptive_optimizer
        self.active_triggers: Dict[str, PipelineTrigger] = {}
        self.quality_gates: Dict[str, Callable] = {}
        self.notification_handlers: Dict[str, Callable] = {}
        
        self._initialize_triggers()
        self._initialize_quality_gates()
        self._initialize_notifications()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load CI/CD configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            logger.info(f"CI/CD configuration loaded from {self.config_path}")
            return config
            
        except Exception as e:
            logger.error(f"Failed to load CI/CD configuration: {str(e)}")
            # Return default configuration
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default CI/CD configuration."""
        return {
            "pipeline": {"name": "default-mlops", "version": "1.0.0"},
            "triggers": {
                "drift_detection": {"enabled": True, "threshold": 0.85},
                "manual": {"enabled": True},
                "git_webhook": {"enabled": True, "branches": ["main", "develop"]}
            },
            "stages": {
                "data_validation": {"enabled": True, "timeout_minutes": 30},
                "model_training": {"enabled": True, "timeout_minutes": 180},
                "model_validation": {"enabled": True, "timeout_minutes": 60},
                "production_deployment": {"enabled": True, "timeout_minutes": 60}
            },
            "quality_gates": {
                "data_quality": {"enabled": True, "min_completeness": 0.95},
                "model_performance": {"enabled": True, "min_accuracy": 0.80}
            }
        }
    
    def _initialize_triggers(self):
        """Initialize pipeline triggers."""
        triggers_config = self.config.get("triggers", {})
        
        for trigger_name, trigger_config in triggers_config.items():
            if trigger_config.get("enabled", False):
                trigger_type = TriggerType(trigger_name)
                
                trigger = PipelineTrigger(
                    trigger_type=trigger_type,
                    enabled=True,
                    conditions=trigger_config,
                    cooldown_hours=trigger_config.get("cooldown_hours")
                )
                
                self.active_triggers[trigger_name] = trigger
                logger.info(f"Initialized trigger: {trigger_name}")
    
    def _initialize_quality_gates(self):
        """Initialize quality gates."""
        self.quality_gates = {
            "data_quality": self._evaluate_data_quality_gate,
            "model_performance": self._evaluate_model_performance_gate,
            "deployment_readiness": self._evaluate_deployment_readiness_gate
        }
        
        logger.info(f"Initialized {len(self.quality_gates)} quality gates")
    
    def _initialize_notifications(self):
        """Initialize notification handlers."""
        self.notification_handlers = {
            "email": self._send_email_notification,
            "slack": self._send_slack_notification,
            "dashboard": self._send_dashboard_notification
        }
        
        logger.info(f"Initialized {len(self.notification_handlers)} notification handlers")
    
    async def evaluate_triggers(self) -> List[str]:
        """Evaluate all active triggers and return triggered pipeline types."""
        triggered_pipelines = []
        
        for trigger_name, trigger in self.active_triggers.items():
            if await self._should_trigger_pipeline(trigger):
                triggered_pipelines.append(trigger_name)
                trigger.last_triggered = datetime.now()
                
                logger.info(f"Pipeline triggered by: {trigger_name}")
        
        return triggered_pipelines
    
    async def _should_trigger_pipeline(self, trigger: PipelineTrigger) -> bool:
        """Determine if a pipeline should be triggered."""
        try:
            # Check cooldown period
            if trigger.cooldown_hours and trigger.last_triggered:
                cooldown_end = trigger.last_triggered + timedelta(hours=trigger.cooldown_hours)
                if datetime.now() < cooldown_end:
                    return False
            
            # Evaluate trigger-specific conditions
            if trigger.trigger_type == TriggerType.DRIFT_DETECTION:
                return await self._evaluate_drift_trigger(trigger.conditions)
            elif trigger.trigger_type == TriggerType.SCHEDULED:
                return await self._evaluate_scheduled_trigger(trigger.conditions)
            elif trigger.trigger_type == TriggerType.DATA_QUALITY:
                return await self._evaluate_data_quality_trigger(trigger.conditions)
            elif trigger.trigger_type == TriggerType.GIT_WEBHOOK:
                return await self._evaluate_git_webhook_trigger(trigger.conditions)
            elif trigger.trigger_type == TriggerType.MANUAL:
                return False  # Manual triggers are handled separately
            
            return False
            
        except Exception as e:
            logger.error(f"Error evaluating trigger {trigger.trigger_type}: {str(e)}")
            return False
    
    async def _evaluate_drift_trigger(self, conditions: Dict[str, Any]) -> bool:
        """Evaluate drift detection trigger conditions."""
        try:
            # Simulate drift detection check
            # In real implementation, this would call the drift detection MCP server
            threshold = conditions.get("threshold", 0.85)
            
            # Mock current accuracy (would be retrieved from monitoring)
            current_accuracy = 0.82  # Simulated value below threshold
            
            return current_accuracy < threshold
            
        except Exception as e:
            logger.error(f"Error evaluating drift trigger: {str(e)}")
            return False
    
    async def _evaluate_scheduled_trigger(self, conditions: Dict[str, Any]) -> bool:
        """Evaluate scheduled trigger conditions."""
        try:
            # Simplified scheduled trigger evaluation
            # In real implementation, this would use cron expressions
            cron = conditions.get("cron", "0 2 * * 0")  # Weekly on Sunday at 2 AM
            
            # For testing, trigger if it's been more than 7 days since last trigger
            # This is a simplified implementation
            return True  # Placeholder
            
        except Exception as e:
            logger.error(f"Error evaluating scheduled trigger: {str(e)}")
            return False
    
    async def _evaluate_data_quality_trigger(self, conditions: Dict[str, Any]) -> bool:
        """Evaluate data quality trigger conditions."""
        try:
            min_completeness = conditions.get("min_completeness", 0.95)
            max_staleness_hours = conditions.get("max_staleness_hours", 24)
            
            # Simulate data quality check
            current_completeness = 0.92  # Below threshold
            data_age_hours = 30  # Above staleness threshold
            
            return (current_completeness < min_completeness or 
                   data_age_hours > max_staleness_hours)
            
        except Exception as e:
            logger.error(f"Error evaluating data quality trigger: {str(e)}")
            return False
    
    async def _evaluate_git_webhook_trigger(self, conditions: Dict[str, Any]) -> bool:
        """Evaluate Git webhook trigger conditions."""
        try:
            # Check if there are recent commits that should trigger pipeline
            current_commit = self.git_integration.get_current_commit()
            if not current_commit:
                return False
            
            # Check if commit should trigger pipeline
            should_trigger, model_types = self.git_integration.should_trigger_pipeline(current_commit)
            
            return should_trigger and len(model_types) > 0
            
        except Exception as e:
            logger.error(f"Error evaluating Git webhook trigger: {str(e)}")
            return False
    
    async def trigger_manual_pipeline(self, model_type: str, config: Dict[str, Any]) -> str:
        """Manually trigger a pipeline run."""
        try:
            # Check if manual triggers are enabled
            manual_trigger = self.active_triggers.get("manual")
            if not manual_trigger or not manual_trigger.enabled:
                raise ValueError("Manual triggers are not enabled")
            
            # Trigger the pipeline
            run_id = await self.mlops_pipeline.trigger_pipeline("manual", model_type, config)
            
            # Send notification
            await self._send_notification("pipeline_start", {
                "model_type": model_type,
                "run_id": run_id,
                "trigger_type": "manual"
            })
            
            logger.info(f"Manual pipeline triggered: {run_id}")
            return run_id
            
        except Exception as e:
            logger.error(f"Failed to trigger manual pipeline: {str(e)}")
            raise
    
    async def handle_git_webhook(self, webhook_payload: Dict[str, Any]) -> List[str]:
        """Handle Git webhook and trigger pipelines as needed."""
        try:
            # Check if Git webhook triggers are enabled
            git_trigger = self.active_triggers.get("git_webhook")
            if not git_trigger or not git_trigger.enabled:
                logger.info("Git webhook triggers are not enabled")
                return []
            
            # Process webhook through Git integration
            triggered_runs = await self.git_integration.handle_webhook(webhook_payload)
            
            # Actually trigger the pipelines
            pipeline_runs = []
            for run_info in triggered_runs:
                # Parse run info: {model_type}_{commit_hash}
                parts = run_info.split('_')
                if len(parts) >= 2:
                    model_type = parts[0]
                    commit_hash = '_'.join(parts[1:])
                    
                    # Get commit info for configuration
                    current_commit = self.git_integration.get_current_commit()
                    if current_commit and current_commit.hash.startswith(commit_hash):
                        config = {
                            "git_commit": current_commit.hash,
                            "git_branch": current_commit.branch,
                            "git_author": current_commit.author,
                            "trigger_source": "git_webhook",
                            "hyperparameters": self.git_integration._get_default_hyperparameters(model_type)
                        }
                        
                        # Trigger pipeline
                        run_id = await self.mlops_pipeline.trigger_pipeline("git_webhook", model_type, config)
                        pipeline_runs.append(run_id)
                        
                        # Send notification
                        await self._send_notification("pipeline_start", {
                            "model_type": model_type,
                            "run_id": run_id,
                            "trigger_type": "git_webhook",
                            "git_commit": current_commit.hash[:8],
                            "git_branch": current_commit.branch
                        })
            
            logger.info(f"Git webhook triggered {len(pipeline_runs)} pipelines")
            return pipeline_runs
            
        except Exception as e:
            logger.error(f"Failed to handle Git webhook: {str(e)}")
            return []
    
    async def optimize_pipeline_parameters(self, model_type: str, base_config: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize pipeline parameters using adaptive optimization."""
        try:
            # Extract hyperparameters from config
            base_hyperparameters = base_config.get('hyperparameters', {})
            
            # Use adaptive optimizer to optimize hyperparameters
            optimized_hyperparameters = await self.adaptive_optimizer.optimize_hyperparameters(
                model_type, base_hyperparameters, base_config
            )
            
            # Update config with optimized hyperparameters
            optimized_config = base_config.copy()
            optimized_config['hyperparameters'] = optimized_hyperparameters
            
            logger.info(f"Optimized hyperparameters for {model_type}: {optimized_hyperparameters}")
            return optimized_config
            
        except Exception as e:
            logger.error(f"Error optimizing pipeline parameters: {str(e)}")
            return base_config
    
    async def provide_performance_feedback(self, model_id: str, model_type: str, 
                                         performance_metrics: Dict[str, float],
                                         hyperparameters: Dict[str, Any],
                                         training_time: float,
                                         deployment_success: bool,
                                         business_impact: float):
        """Provide performance feedback to the adaptive optimizer."""
        try:
            feedback = LearningFeedback(
                model_id=model_id,
                model_type=model_type,
                performance_metrics=performance_metrics,
                hyperparameters=hyperparameters,
                training_time=training_time,
                deployment_success=deployment_success,
                business_impact=business_impact,
                timestamp=datetime.now()
            )
            
            await self.adaptive_optimizer.add_performance_feedback(feedback)
            
            logger.info(f"Performance feedback provided for model {model_id}")
            
        except Exception as e:
            logger.error(f"Error providing performance feedback: {str(e)}")
    
    async def evaluate_quality_gates(self, run_id: str, stage: str) -> List[QualityGateResult]:
        """Evaluate quality gates for a pipeline stage."""
        try:
            results = []
            gates_config = self.config.get("quality_gates", {})
            
            for gate_name, gate_config in gates_config.items():
                if gate_config.get("enabled", False):
                    if gate_name in self.quality_gates:
                        result = await self.quality_gates[gate_name](run_id, stage, gate_config)
                        results.append(result)
            
            logger.info(f"Evaluated {len(results)} quality gates for run {run_id}, stage {stage}")
            return results
            
        except Exception as e:
            logger.error(f"Error evaluating quality gates: {str(e)}")
            return []
    
    async def _evaluate_data_quality_gate(self, run_id: str, stage: str, 
                                        config: Dict[str, Any]) -> QualityGateResult:
        """Evaluate data quality gate."""
        try:
            min_completeness = config.get("min_completeness", 0.95)
            
            # Simulate data quality evaluation
            current_completeness = 0.97  # Mock value
            
            status = QualityGateStatus.PASSED if current_completeness >= min_completeness else QualityGateStatus.FAILED
            
            return QualityGateResult(
                gate_name="data_quality",
                status=status,
                score=current_completeness,
                threshold=min_completeness,
                message=f"Data completeness: {current_completeness:.2%} (threshold: {min_completeness:.2%})",
                details={"completeness": current_completeness, "threshold": min_completeness}
            )
            
        except Exception as e:
            logger.error(f"Error evaluating data quality gate: {str(e)}")
            return QualityGateResult(
                gate_name="data_quality",
                status=QualityGateStatus.FAILED,
                score=0.0,
                threshold=config.get("min_completeness", 0.95),
                message=f"Error evaluating data quality: {str(e)}"
            )
    
    async def _evaluate_model_performance_gate(self, run_id: str, stage: str, 
                                             config: Dict[str, Any]) -> QualityGateResult:
        """Evaluate model performance gate."""
        try:
            min_accuracy = config.get("min_accuracy", 0.80)
            min_improvement = config.get("min_improvement", 0.02)
            
            # Simulate model performance evaluation
            current_accuracy = 0.87  # Mock value
            baseline_accuracy = 0.84  # Mock baseline
            improvement = current_accuracy - baseline_accuracy
            
            meets_accuracy = current_accuracy >= min_accuracy
            meets_improvement = improvement >= min_improvement
            
            status = QualityGateStatus.PASSED if (meets_accuracy and meets_improvement) else QualityGateStatus.FAILED
            
            return QualityGateResult(
                gate_name="model_performance",
                status=status,
                score=current_accuracy,
                threshold=min_accuracy,
                message=f"Model accuracy: {current_accuracy:.2%}, improvement: {improvement:.2%}",
                details={
                    "accuracy": current_accuracy,
                    "baseline_accuracy": baseline_accuracy,
                    "improvement": improvement,
                    "min_accuracy": min_accuracy,
                    "min_improvement": min_improvement
                }
            )
            
        except Exception as e:
            logger.error(f"Error evaluating model performance gate: {str(e)}")
            return QualityGateResult(
                gate_name="model_performance",
                status=QualityGateStatus.FAILED,
                score=0.0,
                threshold=config.get("min_accuracy", 0.80),
                message=f"Error evaluating model performance: {str(e)}"
            )
    
    async def _evaluate_deployment_readiness_gate(self, run_id: str, stage: str, 
                                                config: Dict[str, Any]) -> QualityGateResult:
        """Evaluate deployment readiness gate."""
        try:
            # Simulate deployment readiness checks
            all_tests_passed = True  # Mock value
            security_scan_passed = True  # Mock value
            performance_benchmarks_met = True  # Mock value
            
            all_checks_passed = all([
                all_tests_passed,
                security_scan_passed,
                performance_benchmarks_met
            ])
            
            status = QualityGateStatus.PASSED if all_checks_passed else QualityGateStatus.FAILED
            
            return QualityGateResult(
                gate_name="deployment_readiness",
                status=status,
                score=1.0 if all_checks_passed else 0.0,
                threshold=1.0,
                message=f"Deployment readiness: {'READY' if all_checks_passed else 'NOT READY'}",
                details={
                    "all_tests_passed": all_tests_passed,
                    "security_scan_passed": security_scan_passed,
                    "performance_benchmarks_met": performance_benchmarks_met
                }
            )
            
        except Exception as e:
            logger.error(f"Error evaluating deployment readiness gate: {str(e)}")
            return QualityGateResult(
                gate_name="deployment_readiness",
                status=QualityGateStatus.FAILED,
                score=0.0,
                threshold=1.0,
                message=f"Error evaluating deployment readiness: {str(e)}"
            )
    
    async def _send_notification(self, event_type: str, context: Dict[str, Any]):
        """Send notifications for pipeline events."""
        try:
            notifications_config = self.config.get("notifications", {})
            event_config = notifications_config.get(event_type, {})
            
            if not event_config.get("enabled", False):
                return
            
            channels = event_config.get("channels", [])
            message_template = event_config.get("message", "Pipeline event: {event_type}")
            
            # Format message with context
            message = message_template.format(event_type=event_type, **context)
            
            # Send to each configured channel
            for channel in channels:
                if channel in self.notification_handlers:
                    await self.notification_handlers[channel](message, context)
            
            logger.info(f"Sent {event_type} notification to {len(channels)} channels")
            
        except Exception as e:
            logger.error(f"Error sending notification: {str(e)}")
    
    async def _send_email_notification(self, message: str, context: Dict[str, Any]):
        """Send email notification."""
        try:
            # Simulate email sending
            logger.info(f"EMAIL: {message}")
            
        except Exception as e:
            logger.error(f"Error sending email notification: {str(e)}")
    
    async def _send_slack_notification(self, message: str, context: Dict[str, Any]):
        """Send Slack notification."""
        try:
            # Simulate Slack message sending
            logger.info(f"SLACK: {message}")
            
        except Exception as e:
            logger.error(f"Error sending Slack notification: {str(e)}")
    
    async def _send_dashboard_notification(self, message: str, context: Dict[str, Any]):
        """Send dashboard notification."""
        try:
            # Simulate dashboard notification
            logger.info(f"DASHBOARD: {message}")
            
        except Exception as e:
            logger.error(f"Error sending dashboard notification: {str(e)}")
    
    async def monitor_pipeline_execution(self, run_id: str):
        """Monitor pipeline execution and handle events."""
        try:
            logger.info(f"Starting pipeline monitoring for run: {run_id}")
            
            while True:
                # Get pipeline status
                status = self.mlops_pipeline.get_pipeline_status(run_id)
                
                if not status:
                    logger.warning(f"Pipeline run not found: {run_id}")
                    break
                
                current_status = PipelineStatus(status['status'])
                current_stage = status.get('current_stage')
                
                # Evaluate quality gates for current stage
                if current_stage:
                    quality_results = await self.evaluate_quality_gates(run_id, current_stage)
                    
                    # Check if any quality gates failed
                    failed_gates = [r for r in quality_results if r.status == QualityGateStatus.FAILED]
                    if failed_gates:
                        logger.warning(f"Quality gates failed for run {run_id}: {[g.gate_name for g in failed_gates]}")
                        # Could trigger pipeline cancellation here
                
                # Handle pipeline completion
                if current_status in [PipelineStatus.SUCCESS, PipelineStatus.FAILED, PipelineStatus.CANCELLED]:
                    if current_status == PipelineStatus.SUCCESS:
                        await self._send_notification("pipeline_success", {
                            "run_id": run_id,
                            "model_id": status.get('artifacts', {}).get('model_id', 'unknown')
                        })
                    else:
                        await self._send_notification("pipeline_failure", {
                            "run_id": run_id,
                            "failed_stage": current_stage or 'unknown',
                            "error_message": "Pipeline execution failed"
                        })
                    
                    logger.info(f"Pipeline monitoring completed for run: {run_id}")
                    break
                
                # Wait before next check
                await asyncio.sleep(30)  # Check every 30 seconds
                
        except Exception as e:
            logger.error(f"Error monitoring pipeline execution: {str(e)}")
    
    def get_pipeline_configuration(self) -> Dict[str, Any]:
        """Get current pipeline configuration."""
        return {
            "pipeline_info": self.config.get("pipeline", {}),
            "active_triggers": {
                name: {
                    "type": trigger.trigger_type.value,
                    "enabled": trigger.enabled,
                    "last_triggered": trigger.last_triggered.isoformat() if trigger.last_triggered else None
                }
                for name, trigger in self.active_triggers.items()
            },
            "quality_gates": list(self.quality_gates.keys()),
            "notification_channels": list(self.notification_handlers.keys())
        }
    
    async def update_configuration(self, new_config: Dict[str, Any]):
        """Update pipeline configuration."""
        try:
            # Merge with existing configuration
            self.config.update(new_config)
            
            # Reinitialize components
            self._initialize_triggers()
            self._initialize_quality_gates()
            self._initialize_notifications()
            
            # Save updated configuration
            with open(self.config_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
            
            logger.info("Pipeline configuration updated successfully")
            
        except Exception as e:
            logger.error(f"Error updating configuration: {str(e)}")
            raise


# Global CI/CD integration instance
cicd_integration = CICDIntegration()