"""
Business Scenario Testing Framework

This module provides comprehensive business scenario testing capabilities
for validating the autonomous demand forecasting system under realistic
business conditions and market scenarios.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np
from pathlib import Path
import json
import sqlite3
import random
from enum import Enum

logger = logging.getLogger(__name__)

class ScenarioType(Enum):
    """Types of business scenarios"""
    SEASONAL_SURGE = "seasonal_surge"
    SUPPLY_DISRUPTION = "supply_disruption"
    MARKET_EXPANSION = "market_expansion"
    ECONOMIC_DOWNTURN = "economic_downturn"
    COMPETITIVE_PRESSURE = "competitive_pressure"
    PRODUCT_LIFECYCLE = "product_lifecycle"
    REGULATORY_CHANGE = "regulatory_change"

@dataclass
class BusinessScenarioConfig:
    """Configuration for business scenario testing"""
    scenario_type: ScenarioType
    duration_days: int
    market_impact_severity: float  # 0.0 to 1.0
    affected_product_categories: List[str]
    customer_behavior_changes: Dict[str, float]
    supply_chain_impacts: Dict[str, float]
    competitive_dynamics: Dict[str, float]
    external_factors: Dict[str, Any]

@dataclass
class ScenarioValidationCriteria:
    """Validation criteria for business scenarios"""
    min_forecast_accuracy: float
    max_inventory_waste: float
    min_service_level: float
    max_cost_increase: float
    min_revenue_protection: float
    max_response_time_hours: int
    min_system_availability: float

@dataclass
class BusinessMetrics:
    """Business metrics collected during scenario execution"""
    forecast_accuracy: float
    inventory_turnover: float
    service_level: float
    cost_impact: float
    revenue_impact: float
    customer_satisfaction: float
    operational_efficiency: float
    market_share_change: float

class BusinessScenarioTester:
    """
    Comprehensive business scenario testing framework.
    
    Tests the autonomous demand forecasting system under various realistic
    business conditions to validate system performance and business value.
    """
    
    def __init__(self, results_db: str = "business_scenario_tests.db"):
        self.results_db = results_db
        self.scenario_results: List[Dict[str, Any]] = []
        self.predefined_scenarios: Dict[str, BusinessScenarioConfig] = {}
        
        self._setup_scenario_database()
        self._initialize_predefined_scenarios()
        logger.info("BusinessScenarioTester initialized")
    
    def _setup_scenario_database(self):
        """Initialize database for storing scenario test results"""
        conn = sqlite3.connect(self.results_db)
        cursor = conn.cursor()
        
        cursor.executescript("""
            DROP TABLE IF EXISTS scenario_tests;
            DROP TABLE IF EXISTS scenario_metrics;
            DROP TABLE IF EXISTS scenario_validations;
            DROP TABLE IF EXISTS market_conditions;
            
            CREATE TABLE scenario_tests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                test_id TEXT UNIQUE NOT NULL,
                scenario_name TEXT NOT NULL,
                scenario_type TEXT NOT NULL,
                start_time DATETIME NOT NULL,
                end_time DATETIME,
                duration_days INTEGER,
                status TEXT CHECK(status IN ('RUNNING', 'COMPLETED', 'FAILED', 'TIMEOUT')),
                config_json TEXT,
                results_json TEXT
            );
            
            CREATE TABLE scenario_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                test_id TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL,
                measurement_time DATETIME NOT NULL,
                business_context TEXT,
                FOREIGN KEY (test_id) REFERENCES scenario_tests(test_id)
            );
            
            CREATE TABLE scenario_validations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                test_id TEXT NOT NULL,
                validation_criterion TEXT NOT NULL,
                expected_value REAL,
                actual_value REAL,
                passed BOOLEAN,
                validation_time DATETIME NOT NULL,
                FOREIGN KEY (test_id) REFERENCES scenario_tests(test_id)
            );
            
            CREATE TABLE market_conditions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                test_id TEXT NOT NULL,
                condition_type TEXT NOT NULL,
                condition_value REAL,
                impact_severity REAL,
                duration_hours REAL,
                applied_time DATETIME NOT NULL,
                FOREIGN KEY (test_id) REFERENCES scenario_tests(test_id)
            );
        """)
        
        conn.commit()
        conn.close()
        logger.info("Business scenario testing database initialized")
    
    def _initialize_predefined_scenarios(self):
        """Initialize predefined business scenarios"""
        
        # Black Friday Holiday Surge
        self.predefined_scenarios["black_friday_surge"] = BusinessScenarioConfig(
            scenario_type=ScenarioType.SEASONAL_SURGE,
            duration_days=7,
            market_impact_severity=0.8,
            affected_product_categories=["Electronics", "Clothing", "Home & Garden"],
            customer_behavior_changes={
                "purchase_frequency": 2.5,
                "price_sensitivity": -0.3,
                "brand_switching": 0.4,
                "impulse_buying": 0.6
            },
            supply_chain_impacts={
                "lead_time_increase": 0.2,
                "stockout_probability": 0.15,
                "shipping_delays": 0.3
            },
            competitive_dynamics={
                "price_competition": 0.7,
                "promotional_intensity": 0.9,
                "market_share_volatility": 0.5
            },
            external_factors={
                "media_coverage": 0.8,
                "economic_sentiment": 0.6,
                "weather_impact": 0.1
            }
        )
        
        # Supply Chain Crisis
        self.predefined_scenarios["supply_chain_crisis"] = BusinessScenarioConfig(
            scenario_type=ScenarioType.SUPPLY_DISRUPTION,
            duration_days=30,
            market_impact_severity=0.9,
            affected_product_categories=["Electronics", "Automotive", "Industrial"],
            customer_behavior_changes={
                "purchase_frequency": -0.4,
                "price_sensitivity": 0.5,
                "brand_switching": 0.8,
                "substitute_seeking": 0.7
            },
            supply_chain_impacts={
                "lead_time_increase": 1.5,
                "stockout_probability": 0.4,
                "cost_increase": 0.3,
                "supplier_reliability": -0.6
            },
            competitive_dynamics={
                "price_competition": -0.2,
                "market_consolidation": 0.4,
                "alternative_sourcing": 0.6
            },
            external_factors={
                "regulatory_response": 0.3,
                "media_attention": 0.7,
                "economic_impact": 0.5
            }
        )
        
        # New Market Entry
        self.predefined_scenarios["market_expansion"] = BusinessScenarioConfig(
            scenario_type=ScenarioType.MARKET_EXPANSION,
            duration_days=90,
            market_impact_severity=0.6,
            affected_product_categories=["All"],
            customer_behavior_changes={
                "market_awareness": 0.3,
                "trial_rate": 0.4,
                "adoption_speed": 0.2,
                "loyalty_development": 0.1
            },
            supply_chain_impacts={
                "distribution_expansion": 0.5,
                "inventory_buildup": 0.4,
                "logistics_complexity": 0.3
            },
            competitive_dynamics={
                "competitive_response": 0.6,
                "price_pressure": 0.3,
                "market_share_redistribution": 0.4
            },
            external_factors={
                "regulatory_compliance": 0.4,
                "cultural_adaptation": 0.5,
                "economic_conditions": 0.3
            }
        )
        
        # Economic Recession
        self.predefined_scenarios["economic_recession"] = BusinessScenarioConfig(
            scenario_type=ScenarioType.ECONOMIC_DOWNTURN,
            duration_days=180,
            market_impact_severity=0.7,
            affected_product_categories=["Luxury", "Discretionary", "Premium"],
            customer_behavior_changes={
                "purchase_frequency": -0.4,
                "price_sensitivity": 0.8,
                "brand_downgrading": 0.6,
                "delayed_purchases": 0.5
            },
            supply_chain_impacts={
                "demand_volatility": 0.6,
                "inventory_reduction": 0.3,
                "supplier_consolidation": 0.4
            },
            competitive_dynamics={
                "price_wars": 0.8,
                "market_exit": 0.3,
                "value_positioning": 0.7
            },
            external_factors={
                "unemployment_rate": 0.6,
                "consumer_confidence": -0.7,
                "government_intervention": 0.4
            }
        )
        
        # Competitive Disruption
        self.predefined_scenarios["competitive_disruption"] = BusinessScenarioConfig(
            scenario_type=ScenarioType.COMPETITIVE_PRESSURE,
            duration_days=60,
            market_impact_severity=0.6,
            affected_product_categories=["Technology", "Consumer Goods"],
            customer_behavior_changes={
                "brand_switching": 0.5,
                "price_comparison": 0.7,
                "feature_sensitivity": 0.6,
                "loyalty_erosion": 0.4
            },
            supply_chain_impacts={
                "cost_pressure": 0.4,
                "innovation_acceleration": 0.5,
                "quality_competition": 0.3
            },
            competitive_dynamics={
                "new_entrant_impact": 0.8,
                "price_competition": 0.7,
                "feature_competition": 0.6,
                "marketing_intensity": 0.8
            },
            external_factors={
                "technology_disruption": 0.7,
                "regulatory_changes": 0.2,
                "market_maturity": 0.4
            }
        )
        
        logger.info(f"Initialized {len(self.predefined_scenarios)} predefined business scenarios")
    
    async def run_business_scenario_test(self, scenario_name: str, 
                                       validation_criteria: ScenarioValidationCriteria = None) -> Dict[str, Any]:
        """Run a comprehensive business scenario test"""
        logger.info(f"Starting business scenario test: {scenario_name}")
        
        if scenario_name not in self.predefined_scenarios:
            raise ValueError(f"Unknown scenario: {scenario_name}")
        
        scenario_config = self.predefined_scenarios[scenario_name]
        test_id = f"{scenario_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        start_time = datetime.now()
        
        # Default validation criteria if not provided
        if validation_criteria is None:
            validation_criteria = self._get_default_validation_criteria(scenario_config.scenario_type)
        
        try:
            # Initialize scenario test
            self._initialize_scenario_test(test_id, scenario_name, scenario_config, start_time)
            
            # Execute scenario phases
            scenario_result = await self._execute_scenario_phases(test_id, scenario_config)
            
            # Collect business metrics
            business_metrics = await self._collect_business_metrics(test_id, scenario_config, scenario_result)
            
            # Validate scenario outcomes
            validation_result = self._validate_scenario_outcomes(test_id, business_metrics, validation_criteria)
            
            # Generate scenario report
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            test_result = {
                'test_id': test_id,
                'scenario_name': scenario_name,
                'scenario_type': scenario_config.scenario_type.value,
                'status': 'COMPLETED',
                'execution_time': execution_time,
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'business_metrics': asdict(business_metrics),
                'validation_result': validation_result,
                'scenario_config': asdict(scenario_config),
                'market_impact_analysis': self._analyze_market_impact(scenario_config, business_metrics)
            }
            
            # Save results
            self._save_scenario_test_result(test_id, test_result)
            self.scenario_results.append(test_result)
            
            logger.info(f"Business scenario test completed: {scenario_name} - {test_result['status']}")
            return test_result
            
        except Exception as e:
            logger.error(f"Business scenario test failed: {scenario_name} - {str(e)}")
            
            error_result = {
                'test_id': test_id,
                'scenario_name': scenario_name,
                'status': 'FAILED',
                'error_message': str(e),
                'execution_time': (datetime.now() - start_time).total_seconds()
            }
            
            self._save_scenario_test_result(test_id, error_result)
            return error_result
    
    def _get_default_validation_criteria(self, scenario_type: ScenarioType) -> ScenarioValidationCriteria:
        """Get default validation criteria based on scenario type"""
        
        criteria_map = {
            ScenarioType.SEASONAL_SURGE: ScenarioValidationCriteria(
                min_forecast_accuracy=0.80,
                max_inventory_waste=0.15,
                min_service_level=0.90,
                max_cost_increase=0.20,
                min_revenue_protection=0.85,
                max_response_time_hours=6,
                min_system_availability=0.995
            ),
            ScenarioType.SUPPLY_DISRUPTION: ScenarioValidationCriteria(
                min_forecast_accuracy=0.75,
                max_inventory_waste=0.25,
                min_service_level=0.80,
                max_cost_increase=0.30,
                min_revenue_protection=0.70,
                max_response_time_hours=12,
                min_system_availability=0.99
            ),
            ScenarioType.MARKET_EXPANSION: ScenarioValidationCriteria(
                min_forecast_accuracy=0.70,
                max_inventory_waste=0.20,
                min_service_level=0.85,
                max_cost_increase=0.25,
                min_revenue_protection=0.75,
                max_response_time_hours=24,
                min_system_availability=0.995
            ),
            ScenarioType.ECONOMIC_DOWNTURN: ScenarioValidationCriteria(
                min_forecast_accuracy=0.78,
                max_inventory_waste=0.30,
                min_service_level=0.85,
                max_cost_increase=0.15,
                min_revenue_protection=0.80,
                max_response_time_hours=8,
                min_system_availability=0.99
            ),
            ScenarioType.COMPETITIVE_PRESSURE: ScenarioValidationCriteria(
                min_forecast_accuracy=0.82,
                max_inventory_waste=0.18,
                min_service_level=0.92,
                max_cost_increase=0.12,
                min_revenue_protection=0.88,
                max_response_time_hours=4,
                min_system_availability=0.998
            )
        }
        
        return criteria_map.get(scenario_type, ScenarioValidationCriteria(
            min_forecast_accuracy=0.75,
            max_inventory_waste=0.20,
            min_service_level=0.85,
            max_cost_increase=0.25,
            min_revenue_protection=0.75,
            max_response_time_hours=12,
            min_system_availability=0.99
        ))
    
    def _initialize_scenario_test(self, test_id: str, scenario_name: str, 
                                config: BusinessScenarioConfig, start_time: datetime):
        """Initialize scenario test in database"""
        conn = sqlite3.connect(self.results_db)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO scenario_tests 
            (test_id, scenario_name, scenario_type, start_time, duration_days, status, config_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            test_id,
            scenario_name,
            config.scenario_type.value,
            start_time,
            config.duration_days,
            'RUNNING',
            json.dumps(asdict(config), default=str)
        ))
        
        conn.commit()
        conn.close()
    
    async def _execute_scenario_phases(self, test_id: str, config: BusinessScenarioConfig) -> Dict[str, Any]:
        """Execute the phases of a business scenario"""
        
        phases = [
            "market_condition_setup",
            "customer_behavior_simulation",
            "supply_chain_impact_simulation",
            "competitive_dynamics_simulation",
            "system_response_monitoring",
            "business_impact_assessment"
        ]
        
        phase_results = {}
        
        for phase in phases:
            logger.info(f"Executing phase: {phase}")
            phase_start = datetime.now()
            
            # Simulate phase execution
            phase_result = await self._simulate_scenario_phase(test_id, phase, config)
            
            phase_duration = (datetime.now() - phase_start).total_seconds()
            phase_results[phase] = {
                'result': phase_result,
                'duration': phase_duration,
                'timestamp': phase_start.isoformat()
            }
            
            # Record phase metrics
            self._record_phase_metrics(test_id, phase, phase_result)
            
            # Add realistic delay between phases
            await asyncio.sleep(0.1)
        
        return phase_results
    
    async def _simulate_scenario_phase(self, test_id: str, phase: str, 
                                     config: BusinessScenarioConfig) -> Dict[str, Any]:
        """Simulate execution of a specific scenario phase"""
        
        if phase == "market_condition_setup":
            return await self._simulate_market_conditions(config)
        elif phase == "customer_behavior_simulation":
            return await self._simulate_customer_behavior_changes(config)
        elif phase == "supply_chain_impact_simulation":
            return await self._simulate_supply_chain_impacts(config)
        elif phase == "competitive_dynamics_simulation":
            return await self._simulate_competitive_dynamics(config)
        elif phase == "system_response_monitoring":
            return await self._simulate_system_response(config)
        elif phase == "business_impact_assessment":
            return await self._simulate_business_impact_assessment(config)
        else:
            return {"status": "unknown_phase", "phase": phase}
    
    async def _simulate_market_conditions(self, config: BusinessScenarioConfig) -> Dict[str, Any]:
        """Simulate market condition changes"""
        await asyncio.sleep(0.05)  # Simulate processing time
        
        market_volatility = config.market_impact_severity * random.uniform(0.8, 1.2)
        demand_shift = random.uniform(-0.5, 0.5) * config.market_impact_severity
        
        return {
            "market_volatility": market_volatility,
            "demand_shift": demand_shift,
            "affected_categories": config.affected_product_categories,
            "external_factor_impact": sum(config.external_factors.values()) / len(config.external_factors),
            "market_condition_severity": config.market_impact_severity
        }
    
    async def _simulate_customer_behavior_changes(self, config: BusinessScenarioConfig) -> Dict[str, Any]:
        """Simulate customer behavior changes"""
        await asyncio.sleep(0.05)
        
        behavior_changes = {}
        for behavior, change_factor in config.customer_behavior_changes.items():
            # Add some randomness to the behavior change
            actual_change = change_factor * random.uniform(0.7, 1.3)
            behavior_changes[f"actual_{behavior}"] = actual_change
        
        # Calculate overall customer impact
        positive_changes = sum(max(0, change) for change in behavior_changes.values())
        negative_changes = sum(min(0, change) for change in behavior_changes.values())
        
        return {
            "behavior_changes": behavior_changes,
            "overall_customer_impact": positive_changes + negative_changes,
            "customer_adaptation_rate": random.uniform(0.6, 0.9),
            "behavior_persistence": random.uniform(0.3, 0.8)
        }
    
    async def _simulate_supply_chain_impacts(self, config: BusinessScenarioConfig) -> Dict[str, Any]:
        """Simulate supply chain impacts"""
        await asyncio.sleep(0.05)
        
        supply_impacts = {}
        for impact, severity in config.supply_chain_impacts.items():
            # Simulate actual impact with some variance
            actual_impact = severity * random.uniform(0.8, 1.2)
            supply_impacts[f"actual_{impact}"] = actual_impact
        
        # Calculate supply chain resilience
        total_impact = sum(abs(impact) for impact in supply_impacts.values())
        resilience_score = max(0.1, 1.0 - (total_impact / len(supply_impacts)))
        
        return {
            "supply_impacts": supply_impacts,
            "supply_chain_resilience": resilience_score,
            "recovery_time_estimate": random.uniform(5, 30),  # days
            "alternative_sourcing_success": random.uniform(0.4, 0.8)
        }
    
    async def _simulate_competitive_dynamics(self, config: BusinessScenarioConfig) -> Dict[str, Any]:
        """Simulate competitive dynamics"""
        await asyncio.sleep(0.05)
        
        competitive_responses = {}
        for dynamic, intensity in config.competitive_dynamics.items():
            # Simulate competitive response
            response_strength = intensity * random.uniform(0.6, 1.4)
            competitive_responses[f"response_{dynamic}"] = response_strength
        
        # Calculate market share impact
        total_competitive_pressure = sum(competitive_responses.values())
        market_share_risk = min(0.5, total_competitive_pressure / len(competitive_responses) * 0.3)
        
        return {
            "competitive_responses": competitive_responses,
            "market_share_risk": market_share_risk,
            "competitive_advantage_erosion": random.uniform(0.1, 0.4),
            "strategic_response_effectiveness": random.uniform(0.6, 0.9)
        }
    
    async def _simulate_system_response(self, config: BusinessScenarioConfig) -> Dict[str, Any]:
        """Simulate system response to scenario conditions"""
        await asyncio.sleep(0.1)  # Longer processing for system response
        
        # Simulate system adaptation based on scenario severity
        adaptation_speed = max(0.3, 1.0 - config.market_impact_severity * 0.5)
        response_accuracy = random.uniform(0.7, 0.95) * adaptation_speed
        
        # Simulate different system components' responses
        component_responses = {
            "drift_detection": {
                "detection_speed": random.uniform(0.5, 2.0),  # hours
                "accuracy": random.uniform(0.85, 0.98),
                "false_positive_rate": random.uniform(0.02, 0.08)
            },
            "model_retraining": {
                "trigger_speed": random.uniform(1, 6),  # hours
                "training_success": random.random() > 0.05,
                "accuracy_improvement": random.uniform(0.02, 0.15)
            },
            "inventory_optimization": {
                "rebalancing_speed": random.uniform(2, 12),  # hours
                "optimization_effectiveness": random.uniform(0.7, 0.9),
                "cost_impact": random.uniform(-0.1, 0.2)
            },
            "demand_forecasting": {
                "forecast_adjustment_speed": random.uniform(0.5, 3),  # hours
                "accuracy_under_stress": response_accuracy,
                "confidence_intervals": random.uniform(0.8, 0.95)
            }
        }
        
        return {
            "system_adaptation_speed": adaptation_speed,
            "overall_response_accuracy": response_accuracy,
            "component_responses": component_responses,
            "system_stability": random.uniform(0.85, 0.98),
            "resource_utilization": random.uniform(0.4, 0.8)
        }
    
    async def _simulate_business_impact_assessment(self, config: BusinessScenarioConfig) -> Dict[str, Any]:
        """Simulate business impact assessment"""
        await asyncio.sleep(0.05)
        
        # Calculate business impacts based on scenario type and severity
        base_impact = config.market_impact_severity
        
        if config.scenario_type == ScenarioType.SEASONAL_SURGE:
            revenue_impact = base_impact * random.uniform(1.2, 1.8)
            cost_impact = base_impact * random.uniform(0.1, 0.3)
        elif config.scenario_type == ScenarioType.SUPPLY_DISRUPTION:
            revenue_impact = -base_impact * random.uniform(0.2, 0.6)
            cost_impact = base_impact * random.uniform(0.2, 0.5)
        elif config.scenario_type == ScenarioType.ECONOMIC_DOWNTURN:
            revenue_impact = -base_impact * random.uniform(0.3, 0.7)
            cost_impact = base_impact * random.uniform(0.1, 0.2)
        else:
            revenue_impact = base_impact * random.uniform(-0.3, 0.3)
            cost_impact = base_impact * random.uniform(0.1, 0.3)
        
        return {
            "revenue_impact": revenue_impact,
            "cost_impact": cost_impact,
            "margin_impact": revenue_impact - cost_impact,
            "customer_satisfaction_impact": random.uniform(-0.2, 0.1),
            "operational_efficiency_impact": random.uniform(-0.3, 0.2),
            "market_position_impact": random.uniform(-0.2, 0.2),
            "long_term_value_impact": random.uniform(-0.1, 0.3)
        }
    
    async def _collect_business_metrics(self, test_id: str, config: BusinessScenarioConfig, 
                                      scenario_result: Dict[str, Any]) -> BusinessMetrics:
        """Collect comprehensive business metrics from scenario execution"""
        
        # Extract metrics from scenario results
        system_response = scenario_result.get("system_response_monitoring", {}).get("result", {})
        business_impact = scenario_result.get("business_impact_assessment", {}).get("result", {})
        
        # Calculate forecast accuracy
        forecast_accuracy = system_response.get("overall_response_accuracy", 0.8)
        
        # Calculate inventory turnover (affected by scenario type)
        base_turnover = 1.2
        if config.scenario_type == ScenarioType.SEASONAL_SURGE:
            inventory_turnover = base_turnover * random.uniform(1.5, 2.2)
        elif config.scenario_type == ScenarioType.SUPPLY_DISRUPTION:
            inventory_turnover = base_turnover * random.uniform(0.6, 1.1)
        else:
            inventory_turnover = base_turnover * random.uniform(0.9, 1.4)
        
        # Calculate service level
        service_level = max(0.5, 0.95 - config.market_impact_severity * 0.2 + random.uniform(-0.05, 0.05))
        
        # Extract business impacts
        cost_impact = business_impact.get("cost_impact", 0.1)
        revenue_impact = business_impact.get("revenue_impact", 0.0)
        
        # Calculate customer satisfaction
        customer_satisfaction = max(0.3, 0.85 + business_impact.get("customer_satisfaction_impact", 0.0))
        
        # Calculate operational efficiency
        operational_efficiency = max(0.4, 0.80 + business_impact.get("operational_efficiency_impact", 0.0))
        
        # Calculate market share change
        market_share_change = business_impact.get("market_position_impact", 0.0)
        
        metrics = BusinessMetrics(
            forecast_accuracy=forecast_accuracy,
            inventory_turnover=inventory_turnover,
            service_level=service_level,
            cost_impact=cost_impact,
            revenue_impact=revenue_impact,
            customer_satisfaction=customer_satisfaction,
            operational_efficiency=operational_efficiency,
            market_share_change=market_share_change
        )
        
        # Record metrics in database
        self._record_business_metrics(test_id, metrics)
        
        return metrics
    
    def _validate_scenario_outcomes(self, test_id: str, metrics: BusinessMetrics, 
                                  criteria: ScenarioValidationCriteria) -> Dict[str, Any]:
        """Validate scenario outcomes against business criteria"""
        
        validations = {}
        passed_count = 0
        total_count = 0
        
        # Validate forecast accuracy
        validations["forecast_accuracy"] = {
            "criterion": "min_forecast_accuracy",
            "expected": criteria.min_forecast_accuracy,
            "actual": metrics.forecast_accuracy,
            "passed": metrics.forecast_accuracy >= criteria.min_forecast_accuracy
        }
        
        # Validate inventory waste (inverse of turnover efficiency)
        inventory_waste = max(0, 1.0 - (metrics.inventory_turnover / 2.0))
        validations["inventory_waste"] = {
            "criterion": "max_inventory_waste",
            "expected": criteria.max_inventory_waste,
            "actual": inventory_waste,
            "passed": inventory_waste <= criteria.max_inventory_waste
        }
        
        # Validate service level
        validations["service_level"] = {
            "criterion": "min_service_level",
            "expected": criteria.min_service_level,
            "actual": metrics.service_level,
            "passed": metrics.service_level >= criteria.min_service_level
        }
        
        # Validate cost impact
        validations["cost_impact"] = {
            "criterion": "max_cost_increase",
            "expected": criteria.max_cost_increase,
            "actual": metrics.cost_impact,
            "passed": metrics.cost_impact <= criteria.max_cost_increase
        }
        
        # Validate revenue protection (for negative revenue impacts)
        revenue_protection = max(0, 1.0 + metrics.revenue_impact) if metrics.revenue_impact < 0 else 1.0
        validations["revenue_protection"] = {
            "criterion": "min_revenue_protection",
            "expected": criteria.min_revenue_protection,
            "actual": revenue_protection,
            "passed": revenue_protection >= criteria.min_revenue_protection
        }
        
        # Count passed validations
        for validation in validations.values():
            total_count += 1
            if validation["passed"]:
                passed_count += 1
        
        # Record validations in database
        self._record_validations(test_id, validations)
        
        return {
            "validations": validations,
            "passed_count": passed_count,
            "total_count": total_count,
            "success_rate": passed_count / total_count if total_count > 0 else 0,
            "overall_passed": passed_count == total_count
        }
    
    def _analyze_market_impact(self, config: BusinessScenarioConfig, 
                             metrics: BusinessMetrics) -> Dict[str, Any]:
        """Analyze market impact of the scenario"""
        
        return {
            "scenario_severity": config.market_impact_severity,
            "business_resilience": min(1.0, metrics.operational_efficiency + 0.2),
            "market_adaptation": max(0.0, 1.0 - abs(metrics.market_share_change)),
            "financial_impact": metrics.revenue_impact - metrics.cost_impact,
            "customer_impact": metrics.customer_satisfaction,
            "operational_impact": metrics.operational_efficiency,
            "strategic_implications": {
                "market_position": "strengthened" if metrics.market_share_change > 0.05 else "maintained" if metrics.market_share_change > -0.05 else "weakened",
                "competitive_advantage": "improved" if metrics.forecast_accuracy > 0.85 else "maintained",
                "operational_excellence": "high" if metrics.operational_efficiency > 0.85 else "medium" if metrics.operational_efficiency > 0.70 else "low"
            }
        }
    
    def _record_phase_metrics(self, test_id: str, phase: str, phase_result: Dict[str, Any]):
        """Record phase-specific metrics"""
        conn = sqlite3.connect(self.results_db)
        cursor = conn.cursor()
        
        timestamp = datetime.now()
        
        # Record key metrics from phase result
        for key, value in phase_result.items():
            if isinstance(value, (int, float)):
                cursor.execute("""
                    INSERT INTO scenario_metrics 
                    (test_id, metric_name, metric_value, measurement_time, business_context)
                    VALUES (?, ?, ?, ?, ?)
                """, (test_id, f"{phase}_{key}", value, timestamp, phase))
        
        conn.commit()
        conn.close()
    
    def _record_business_metrics(self, test_id: str, metrics: BusinessMetrics):
        """Record business metrics in database"""
        conn = sqlite3.connect(self.results_db)
        cursor = conn.cursor()
        
        timestamp = datetime.now()
        
        for metric_name, metric_value in asdict(metrics).items():
            cursor.execute("""
                INSERT INTO scenario_metrics 
                (test_id, metric_name, metric_value, measurement_time, business_context)
                VALUES (?, ?, ?, ?, ?)
            """, (test_id, metric_name, metric_value, timestamp, "business_outcome"))
        
        conn.commit()
        conn.close()
    
    def _record_validations(self, test_id: str, validations: Dict[str, Any]):
        """Record validation results in database"""
        conn = sqlite3.connect(self.results_db)
        cursor = conn.cursor()
        
        timestamp = datetime.now()
        
        for validation_name, validation_data in validations.items():
            cursor.execute("""
                INSERT INTO scenario_validations 
                (test_id, validation_criterion, expected_value, actual_value, passed, validation_time)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                test_id,
                validation_data["criterion"],
                validation_data["expected"],
                validation_data["actual"],
                validation_data["passed"],
                timestamp
            ))
        
        conn.commit()
        conn.close()
    
    def _save_scenario_test_result(self, test_id: str, result: Dict[str, Any]):
        """Save complete scenario test result"""
        conn = sqlite3.connect(self.results_db)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE scenario_tests 
            SET end_time = ?, status = ?, results_json = ?
            WHERE test_id = ?
        """, (
            datetime.now(),
            result["status"],
            json.dumps(result, default=str),
            test_id
        ))
        
        conn.commit()
        conn.close()
    
    async def run_all_predefined_scenarios(self) -> Dict[str, Any]:
        """Run all predefined business scenarios"""
        logger.info("Running all predefined business scenarios")
        
        results = {
            "test_summary": {
                "total_scenarios": len(self.predefined_scenarios),
                "start_time": datetime.now().isoformat()
            },
            "scenario_results": [],
            "overall_metrics": {}
        }
        
        # Run scenarios concurrently
        scenario_tasks = []
        for scenario_name in self.predefined_scenarios.keys():
            task = asyncio.create_task(self.run_business_scenario_test(scenario_name))
            scenario_tasks.append((scenario_name, task))
        
        # Collect results
        passed_scenarios = 0
        failed_scenarios = 0
        
        for scenario_name, task in scenario_tasks:
            try:
                result = await task
                results["scenario_results"].append(result)
                
                if result.get("status") == "COMPLETED":
                    validation_result = result.get("validation_result", {})
                    if validation_result.get("overall_passed", False):
                        passed_scenarios += 1
                    else:
                        failed_scenarios += 1
                else:
                    failed_scenarios += 1
                    
            except Exception as e:
                logger.error(f"Scenario {scenario_name} failed: {e}")
                failed_scenarios += 1
                results["scenario_results"].append({
                    "scenario_name": scenario_name,
                    "status": "ERROR",
                    "error": str(e)
                })
        
        # Calculate overall metrics
        results["test_summary"].update({
            "end_time": datetime.now().isoformat(),
            "passed_scenarios": passed_scenarios,
            "failed_scenarios": failed_scenarios,
            "success_rate": passed_scenarios / len(self.predefined_scenarios)
        })
        
        results["overall_metrics"] = self._calculate_overall_business_metrics(results["scenario_results"])
        
        logger.info(f"All scenarios completed: {passed_scenarios}/{len(self.predefined_scenarios)} passed")
        return results
    
    def _calculate_overall_business_metrics(self, scenario_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate overall business metrics across all scenarios"""
        
        completed_scenarios = [r for r in scenario_results if r.get("status") == "COMPLETED"]
        
        if not completed_scenarios:
            return {"error": "No completed scenarios to analyze"}
        
        # Aggregate metrics
        total_forecast_accuracy = 0
        total_service_level = 0
        total_operational_efficiency = 0
        total_customer_satisfaction = 0
        
        for result in completed_scenarios:
            business_metrics = result.get("business_metrics", {})
            total_forecast_accuracy += business_metrics.get("forecast_accuracy", 0)
            total_service_level += business_metrics.get("service_level", 0)
            total_operational_efficiency += business_metrics.get("operational_efficiency", 0)
            total_customer_satisfaction += business_metrics.get("customer_satisfaction", 0)
        
        count = len(completed_scenarios)
        
        return {
            "average_forecast_accuracy": total_forecast_accuracy / count,
            "average_service_level": total_service_level / count,
            "average_operational_efficiency": total_operational_efficiency / count,
            "average_customer_satisfaction": total_customer_satisfaction / count,
            "business_resilience_score": (
                (total_forecast_accuracy + total_service_level + total_operational_efficiency + total_customer_satisfaction) 
                / (4 * count)
            ),
            "scenarios_analyzed": count
        }
    
    def generate_business_scenario_report(self, output_file: str = "business_scenario_report.json") -> Dict[str, Any]:
        """Generate comprehensive business scenario testing report"""
        logger.info("Generating business scenario testing report")
        
        report = {
            "report_summary": {
                "generation_time": datetime.now().isoformat(),
                "total_scenarios_tested": len(self.scenario_results),
                "predefined_scenarios": list(self.predefined_scenarios.keys())
            },
            "scenario_results": self.scenario_results,
            "business_impact_analysis": {},
            "validation_summary": {},
            "recommendations": []
        }
        
        if self.scenario_results:
            # Analyze business impacts
            completed_results = [r for r in self.scenario_results if r.get("status") == "COMPLETED"]
            
            if completed_results:
                report["business_impact_analysis"] = self._calculate_overall_business_metrics(completed_results)
                
                # Validation summary
                total_validations = 0
                passed_validations = 0
                
                for result in completed_results:
                    validation_result = result.get("validation_result", {})
                    total_validations += validation_result.get("total_count", 0)
                    passed_validations += validation_result.get("passed_count", 0)
                
                report["validation_summary"] = {
                    "total_validations": total_validations,
                    "passed_validations": passed_validations,
                    "validation_success_rate": passed_validations / total_validations if total_validations > 0 else 0,
                    "scenarios_passed": len([r for r in completed_results if r.get("validation_result", {}).get("overall_passed", False)]),
                    "scenarios_failed": len([r for r in completed_results if not r.get("validation_result", {}).get("overall_passed", True)])
                }
                
                # Generate recommendations
                avg_forecast_accuracy = report["business_impact_analysis"].get("average_forecast_accuracy", 0)
                if avg_forecast_accuracy < 0.80:
                    report["recommendations"].append("Improve forecast accuracy - average below 80%")
                
                avg_service_level = report["business_impact_analysis"].get("average_service_level", 0)
                if avg_service_level < 0.85:
                    report["recommendations"].append("Enhance service level - average below 85%")
                
                validation_rate = report["validation_summary"]["validation_success_rate"]
                if validation_rate < 0.80:
                    report["recommendations"].append("Address validation failures - success rate below 80%")
        
        # Save report
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Business scenario testing report saved to {output_file}")
        return report
    
    def cleanup(self):
        """Clean up testing resources"""
        if Path(self.results_db).exists():
            Path(self.results_db).unlink()
        logger.info("Business scenario tester cleanup completed")