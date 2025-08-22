"""
Adaptive Optimization and Self-Learning Module for MLOps Pipeline.

This module implements self-learning optimization algorithms for retraining strategies,
feedback loops for continuous improvement, and automated parameter tuning.
"""

import asyncio
import json
import logging
import os
import sqlite3
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import pickle
from collections import defaultdict, deque
import statistics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DATABASE_PATH = os.path.join(os.path.dirname(__file__), "database", "autonomous_demand_forecasting.db")


class OptimizationStrategy(Enum):
    """Optimization strategy types."""
    BAYESIAN = "bayesian"
    GENETIC = "genetic"
    GRADIENT_BASED = "gradient_based"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    ENSEMBLE = "ensemble"


class LearningPhase(Enum):
    """Learning phase stages."""
    EXPLORATION = "exploration"
    EXPLOITATION = "exploitation"
    ADAPTATION = "adaptation"


@dataclass
class OptimizationResult:
    """Result of an optimization attempt."""
    strategy: OptimizationStrategy
    parameters: Dict[str, Any]
    performance_score: float
    improvement_over_baseline: float
    execution_time: float
    timestamp: datetime
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class LearningFeedback:
    """Feedback from model performance for learning."""
    model_id: str
    model_type: str
    performance_metrics: Dict[str, float]
    hyperparameters: Dict[str, Any]
    training_time: float
    deployment_success: bool
    business_impact: float
    timestamp: datetime
    context: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.context is None:
            self.context = {}


@dataclass
class AdaptationRule:
    """Rule for adaptive parameter adjustment."""
    rule_id: str
    condition: str  # Python expression to evaluate
    action: str     # Action to take when condition is met
    priority: int   # Higher priority rules are applied first
    enabled: bool = True
    success_count: int = 0
    failure_count: int = 0
    last_applied: Optional[datetime] = None


class PerformancePredictor:
    """Predicts model performance based on hyperparameters and context."""
    
    def __init__(self):
        self.training_history: List[LearningFeedback] = []
        self.feature_importance: Dict[str, float] = {}
        self.performance_models: Dict[str, Any] = {}
        
    def add_training_feedback(self, feedback: LearningFeedback):
        """Add training feedback to the predictor."""
        self.training_history.append(feedback)
        
        # Keep only recent history to avoid memory issues
        if len(self.training_history) > 1000:
            self.training_history = self.training_history[-800:]
        
        # Update feature importance
        self._update_feature_importance(feedback)
    
    def _update_feature_importance(self, feedback: LearningFeedback):
        """Update feature importance based on feedback."""
        try:
            # Simple correlation-based feature importance
            for param_name, param_value in feedback.hyperparameters.items():
                if isinstance(param_value, (int, float)):
                    # Calculate correlation with performance
                    param_values = []
                    performance_values = []
                    
                    for hist_feedback in self.training_history[-50:]:  # Last 50 entries
                        if param_name in hist_feedback.hyperparameters:
                            hist_value = hist_feedback.hyperparameters[param_name]
                            if isinstance(hist_value, (int, float)):
                                param_values.append(hist_value)
                                performance_values.append(
                                    hist_feedback.performance_metrics.get('accuracy', 0.0)
                                )
                    
                    if len(param_values) > 3:  # Need at least 4 data points
                        correlation = np.corrcoef(param_values, performance_values)[0, 1]
                        if not np.isnan(correlation):
                            self.feature_importance[param_name] = abs(correlation)
                            
        except Exception as e:
            logger.error(f"Error updating feature importance: {str(e)}")
    
    def predict_performance(self, model_type: str, hyperparameters: Dict[str, Any], 
                          context: Dict[str, Any] = None) -> float:
        """Predict model performance for given hyperparameters."""
        try:
            if not self.training_history:
                return 0.5  # Default prediction
            
            # Find similar configurations
            similar_configs = []
            for feedback in self.training_history:
                if feedback.model_type == model_type:
                    similarity = self._calculate_similarity(hyperparameters, feedback.hyperparameters)
                    if similarity > 0.3:  # Threshold for similarity
                        similar_configs.append((similarity, feedback))
            
            if not similar_configs:
                # Use average performance for this model type
                type_performances = [
                    f.performance_metrics.get('accuracy', 0.0)
                    for f in self.training_history
                    if f.model_type == model_type
                ]
                return statistics.mean(type_performances) if type_performances else 0.5
            
            # Weighted average based on similarity
            total_weight = 0
            weighted_performance = 0
            
            for similarity, feedback in similar_configs:
                weight = similarity
                performance = feedback.performance_metrics.get('accuracy', 0.0)
                weighted_performance += weight * performance
                total_weight += weight
            
            return weighted_performance / total_weight if total_weight > 0 else 0.5
            
        except Exception as e:
            logger.error(f"Error predicting performance: {str(e)}")
            return 0.5
    
    def _calculate_similarity(self, params1: Dict[str, Any], params2: Dict[str, Any]) -> float:
        """Calculate similarity between two parameter sets."""
        try:
            common_params = set(params1.keys()) & set(params2.keys())
            if not common_params:
                return 0.0
            
            similarities = []
            for param in common_params:
                val1, val2 = params1[param], params2[param]
                
                if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                    # Numerical similarity
                    if val1 == val2:
                        similarities.append(1.0)
                    else:
                        max_val = max(abs(val1), abs(val2), 1e-6)
                        similarity = 1.0 - abs(val1 - val2) / max_val
                        similarities.append(max(0.0, similarity))
                elif val1 == val2:
                    # Exact match for categorical
                    similarities.append(1.0)
                else:
                    # Different categorical values
                    similarities.append(0.0)
            
            return statistics.mean(similarities) if similarities else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {str(e)}")
            return 0.0


class AdaptiveOptimizer:
    """Adaptive optimization system for hyperparameter tuning and strategy selection."""
    
    def __init__(self):
        self.db_path = DATABASE_PATH
        self.performance_predictor = PerformancePredictor()
        self.optimization_history: List[OptimizationResult] = []
        self.adaptation_rules: List[AdaptationRule] = []
        self.strategy_performance: Dict[OptimizationStrategy, deque] = defaultdict(
            lambda: deque(maxlen=50)
        )
        self.current_phase = LearningPhase.EXPLORATION
        self.exploration_count = 0
        self.exploitation_count = 0
        
        self._initialize_adaptation_rules()
    
    def get_db_connection(self):
        """Get database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def _initialize_adaptation_rules(self):
        """Initialize default adaptation rules."""
        self.adaptation_rules = [
            AdaptationRule(
                rule_id="low_accuracy_increase_complexity",
                condition="performance_metrics.get('accuracy', 0) < 0.8",
                action="increase_model_complexity",
                priority=10
            ),
            AdaptationRule(
                rule_id="high_accuracy_reduce_overfitting",
                condition="performance_metrics.get('accuracy', 0) > 0.95",
                action="reduce_overfitting_risk",
                priority=8
            ),
            AdaptationRule(
                rule_id="slow_training_optimize_speed",
                condition="training_time > 3600",  # More than 1 hour
                action="optimize_training_speed",
                priority=6
            ),
            AdaptationRule(
                rule_id="poor_business_impact_adjust_focus",
                condition="business_impact < 0.02",  # Less than 2% improvement
                action="adjust_business_focus",
                priority=9
            ),
            AdaptationRule(
                rule_id="frequent_failures_simplify",
                condition="deployment_success == False",
                action="simplify_model_configuration",
                priority=7
            )
        ]
    
    async def optimize_hyperparameters(self, model_type: str, base_parameters: Dict[str, Any],
                                     context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Optimize hyperparameters using adaptive strategies."""
        try:
            logger.info(f"Starting adaptive optimization for {model_type}")
            
            # Determine optimization strategy based on current phase and history
            strategy = self._select_optimization_strategy(model_type, context)
            
            # Perform optimization
            if strategy == OptimizationStrategy.BAYESIAN:
                optimized_params = await self._bayesian_optimization(model_type, base_parameters, context)
            elif strategy == OptimizationStrategy.GENETIC:
                optimized_params = await self._genetic_optimization(model_type, base_parameters, context)
            elif strategy == OptimizationStrategy.GRADIENT_BASED:
                optimized_params = await self._gradient_based_optimization(model_type, base_parameters, context)
            elif strategy == OptimizationStrategy.REINFORCEMENT_LEARNING:
                optimized_params = await self._reinforcement_learning_optimization(model_type, base_parameters, context)
            else:  # ENSEMBLE
                optimized_params = await self._ensemble_optimization(model_type, base_parameters, context)
            
            # Predict performance of optimized parameters
            predicted_performance = self.performance_predictor.predict_performance(
                model_type, optimized_params, context
            )
            
            # Record optimization attempt
            baseline_performance = self.performance_predictor.predict_performance(
                model_type, base_parameters, context
            )
            
            optimization_result = OptimizationResult(
                strategy=strategy,
                parameters=optimized_params,
                performance_score=predicted_performance,
                improvement_over_baseline=predicted_performance - baseline_performance,
                execution_time=0.0,  # Would be measured in real implementation
                timestamp=datetime.now(),
                metadata={
                    "model_type": model_type,
                    "base_parameters": base_parameters,
                    "context": context or {}
                }
            )
            
            self.optimization_history.append(optimization_result)
            self.strategy_performance[strategy].append(predicted_performance)
            
            # Update learning phase
            self._update_learning_phase()
            
            logger.info(f"Optimization completed with {strategy.value} strategy. "
                       f"Predicted improvement: {optimization_result.improvement_over_baseline:.3f}")
            
            return optimized_params
            
        except Exception as e:
            logger.error(f"Error in adaptive optimization: {str(e)}")
            return base_parameters
    
    def _select_optimization_strategy(self, model_type: str, context: Dict[str, Any] = None) -> OptimizationStrategy:
        """Select the best optimization strategy based on current state."""
        try:
            # In exploration phase, try different strategies
            if self.current_phase == LearningPhase.EXPLORATION:
                # Cycle through strategies to gather data
                strategies = list(OptimizationStrategy)
                return strategies[self.exploration_count % len(strategies)]
            
            # In exploitation phase, use the best performing strategy
            elif self.current_phase == LearningPhase.EXPLOITATION:
                if not self.strategy_performance:
                    return OptimizationStrategy.BAYESIAN  # Default
                
                # Find strategy with highest average performance
                best_strategy = OptimizationStrategy.BAYESIAN
                best_avg_performance = 0.0
                
                for strategy, performances in self.strategy_performance.items():
                    if performances:
                        avg_performance = statistics.mean(performances)
                        if avg_performance > best_avg_performance:
                            best_avg_performance = avg_performance
                            best_strategy = strategy
                
                return best_strategy
            
            # In adaptation phase, use ensemble or context-specific strategy
            else:  # ADAPTATION
                # Use ensemble approach or select based on context
                if context and context.get('complexity', 'medium') == 'high':
                    return OptimizationStrategy.GENETIC
                elif context and context.get('speed_priority', False):
                    return OptimizationStrategy.GRADIENT_BASED
                else:
                    return OptimizationStrategy.ENSEMBLE
                    
        except Exception as e:
            logger.error(f"Error selecting optimization strategy: {str(e)}")
            return OptimizationStrategy.BAYESIAN
    
    async def _bayesian_optimization(self, model_type: str, base_parameters: Dict[str, Any],
                                   context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Bayesian optimization implementation."""
        try:
            # Simplified Bayesian optimization
            optimized_params = base_parameters.copy()
            
            # Adjust parameters based on feature importance
            for param_name, param_value in base_parameters.items():
                if isinstance(param_value, (int, float)):
                    importance = self.performance_predictor.feature_importance.get(param_name, 0.5)
                    
                    # Adjust parameter based on importance and historical performance
                    if importance > 0.7:  # High importance
                        # Make larger adjustments for important parameters
                        adjustment_factor = 0.2
                    elif importance > 0.3:  # Medium importance
                        adjustment_factor = 0.1
                    else:  # Low importance
                        adjustment_factor = 0.05
                    
                    # Random adjustment within bounds
                    adjustment = np.random.normal(0, adjustment_factor * abs(param_value))
                    new_value = param_value + adjustment
                    
                    # Apply bounds based on parameter type
                    if param_name in ['n_estimators', 'max_depth']:
                        new_value = max(1, int(new_value))
                    elif param_name in ['learning_rate', 'subsample']:
                        new_value = max(0.01, min(1.0, new_value))
                    
                    optimized_params[param_name] = new_value
            
            return optimized_params
            
        except Exception as e:
            logger.error(f"Error in Bayesian optimization: {str(e)}")
            return base_parameters
    
    async def _genetic_optimization(self, model_type: str, base_parameters: Dict[str, Any],
                                  context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Genetic algorithm optimization implementation."""
        try:
            # Simplified genetic algorithm
            population_size = 10
            generations = 5
            
            # Create initial population
            population = []
            for _ in range(population_size):
                individual = base_parameters.copy()
                for param_name, param_value in individual.items():
                    if isinstance(param_value, (int, float)):
                        # Mutate parameter
                        mutation_rate = 0.3
                        if np.random.random() < mutation_rate:
                            if param_name in ['n_estimators']:
                                individual[param_name] = max(10, int(param_value * np.random.uniform(0.5, 2.0)))
                            elif param_name in ['max_depth']:
                                individual[param_name] = max(1, int(param_value * np.random.uniform(0.7, 1.5)))
                            elif param_name in ['learning_rate']:
                                individual[param_name] = max(0.001, min(1.0, param_value * np.random.uniform(0.5, 2.0)))
                
                population.append(individual)
            
            # Evolve population
            for generation in range(generations):
                # Evaluate fitness (predicted performance)
                fitness_scores = []
                for individual in population:
                    fitness = self.performance_predictor.predict_performance(model_type, individual, context)
                    fitness_scores.append(fitness)
                
                # Select best individuals
                sorted_indices = np.argsort(fitness_scores)[::-1]  # Descending order
                elite_size = population_size // 2
                elite_population = [population[i] for i in sorted_indices[:elite_size]]
                
                # Create new generation
                new_population = elite_population.copy()
                while len(new_population) < population_size:
                    # Crossover
                    parent1 = np.random.choice(elite_population)
                    parent2 = np.random.choice(elite_population)
                    child = self._crossover(parent1, parent2)
                    new_population.append(child)
                
                population = new_population
            
            # Return best individual
            final_fitness = [
                self.performance_predictor.predict_performance(model_type, individual, context)
                for individual in population
            ]
            best_index = np.argmax(final_fitness)
            return population[best_index]
            
        except Exception as e:
            logger.error(f"Error in genetic optimization: {str(e)}")
            return base_parameters
    
    def _crossover(self, parent1: Dict[str, Any], parent2: Dict[str, Any]) -> Dict[str, Any]:
        """Perform crossover between two parameter sets."""
        child = {}
        for param_name in parent1.keys():
            if np.random.random() < 0.5:
                child[param_name] = parent1[param_name]
            else:
                child[param_name] = parent2[param_name]
        return child
    
    async def _gradient_based_optimization(self, model_type: str, base_parameters: Dict[str, Any],
                                         context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Gradient-based optimization implementation."""
        try:
            # Simplified gradient-based optimization
            optimized_params = base_parameters.copy()
            learning_rate = 0.1
            
            for param_name, param_value in base_parameters.items():
                if isinstance(param_value, (int, float)):
                    # Estimate gradient using finite differences
                    epsilon = 0.01 * abs(param_value) if param_value != 0 else 0.01
                    
                    # Forward difference
                    params_plus = base_parameters.copy()
                    params_plus[param_name] = param_value + epsilon
                    performance_plus = self.performance_predictor.predict_performance(
                        model_type, params_plus, context
                    )
                    
                    # Backward difference
                    params_minus = base_parameters.copy()
                    params_minus[param_name] = param_value - epsilon
                    performance_minus = self.performance_predictor.predict_performance(
                        model_type, params_minus, context
                    )
                    
                    # Calculate gradient
                    gradient = (performance_plus - performance_minus) / (2 * epsilon)
                    
                    # Update parameter
                    new_value = param_value + learning_rate * gradient * abs(param_value)
                    
                    # Apply bounds
                    if param_name in ['n_estimators']:
                        new_value = max(10, int(new_value))
                    elif param_name in ['max_depth']:
                        new_value = max(1, int(new_value))
                    elif param_name in ['learning_rate', 'subsample']:
                        new_value = max(0.001, min(1.0, new_value))
                    
                    optimized_params[param_name] = new_value
            
            return optimized_params
            
        except Exception as e:
            logger.error(f"Error in gradient-based optimization: {str(e)}")
            return base_parameters
    
    async def _reinforcement_learning_optimization(self, model_type: str, base_parameters: Dict[str, Any],
                                                 context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Reinforcement learning-based optimization implementation."""
        try:
            # Simplified Q-learning approach
            optimized_params = base_parameters.copy()
            
            # Define actions (parameter adjustments)
            actions = ['increase', 'decrease', 'no_change']
            
            for param_name, param_value in base_parameters.items():
                if isinstance(param_value, (int, float)):
                    # Select action based on historical performance
                    # This is a simplified implementation
                    action_rewards = {'increase': 0.0, 'decrease': 0.0, 'no_change': 0.0}
                    
                    # Calculate rewards based on optimization history
                    for result in self.optimization_history[-20:]:  # Last 20 results
                        if param_name in result.parameters:
                            original_value = result.metadata.get('base_parameters', {}).get(param_name, param_value)
                            optimized_value = result.parameters[param_name]
                            
                            if optimized_value > original_value:
                                action_rewards['increase'] += result.improvement_over_baseline
                            elif optimized_value < original_value:
                                action_rewards['decrease'] += result.improvement_over_baseline
                            else:
                                action_rewards['no_change'] += result.improvement_over_baseline
                    
                    # Select best action
                    best_action = max(action_rewards, key=action_rewards.get)
                    
                    # Apply action
                    if best_action == 'increase':
                        adjustment_factor = 1.1
                    elif best_action == 'decrease':
                        adjustment_factor = 0.9
                    else:
                        adjustment_factor = 1.0
                    
                    new_value = param_value * adjustment_factor
                    
                    # Apply bounds
                    if param_name in ['n_estimators']:
                        new_value = max(10, int(new_value))
                    elif param_name in ['max_depth']:
                        new_value = max(1, int(new_value))
                    elif param_name in ['learning_rate', 'subsample']:
                        new_value = max(0.001, min(1.0, new_value))
                    
                    optimized_params[param_name] = new_value
            
            return optimized_params
            
        except Exception as e:
            logger.error(f"Error in reinforcement learning optimization: {str(e)}")
            return base_parameters
    
    async def _ensemble_optimization(self, model_type: str, base_parameters: Dict[str, Any],
                                   context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Ensemble optimization using multiple strategies."""
        try:
            # Run multiple optimization strategies
            strategies = [
                OptimizationStrategy.BAYESIAN,
                OptimizationStrategy.GENETIC,
                OptimizationStrategy.GRADIENT_BASED
            ]
            
            results = []
            for strategy in strategies:
                if strategy == OptimizationStrategy.BAYESIAN:
                    result = await self._bayesian_optimization(model_type, base_parameters, context)
                elif strategy == OptimizationStrategy.GENETIC:
                    result = await self._genetic_optimization(model_type, base_parameters, context)
                else:  # GRADIENT_BASED
                    result = await self._gradient_based_optimization(model_type, base_parameters, context)
                
                # Predict performance
                performance = self.performance_predictor.predict_performance(model_type, result, context)
                results.append((result, performance))
            
            # Select best result
            best_result = max(results, key=lambda x: x[1])
            return best_result[0]
            
        except Exception as e:
            logger.error(f"Error in ensemble optimization: {str(e)}")
            return base_parameters
    
    def _update_learning_phase(self):
        """Update the current learning phase based on optimization history."""
        try:
            total_optimizations = len(self.optimization_history)
            
            if total_optimizations < 20:
                self.current_phase = LearningPhase.EXPLORATION
                self.exploration_count += 1
            elif total_optimizations < 100:
                self.current_phase = LearningPhase.EXPLOITATION
                self.exploitation_count += 1
            else:
                self.current_phase = LearningPhase.ADAPTATION
            
            logger.debug(f"Learning phase updated to: {self.current_phase.value}")
            
        except Exception as e:
            logger.error(f"Error updating learning phase: {str(e)}")
    
    async def add_performance_feedback(self, feedback: LearningFeedback):
        """Add performance feedback to improve future optimizations."""
        try:
            # Add to performance predictor
            self.performance_predictor.add_training_feedback(feedback)
            
            # Apply adaptation rules
            await self._apply_adaptation_rules(feedback)
            
            # Store feedback in database
            await self._store_feedback(feedback)
            
            logger.info(f"Performance feedback added for model {feedback.model_id}")
            
        except Exception as e:
            logger.error(f"Error adding performance feedback: {str(e)}")
    
    async def _apply_adaptation_rules(self, feedback: LearningFeedback):
        """Apply adaptation rules based on feedback."""
        try:
            # Sort rules by priority
            sorted_rules = sorted(self.adaptation_rules, key=lambda r: r.priority, reverse=True)
            
            for rule in sorted_rules:
                if not rule.enabled:
                    continue
                
                try:
                    # Evaluate condition
                    context = {
                        'performance_metrics': feedback.performance_metrics,
                        'training_time': feedback.training_time,
                        'deployment_success': feedback.deployment_success,
                        'business_impact': feedback.business_impact,
                        'model_type': feedback.model_type
                    }
                    
                    if eval(rule.condition, {"__builtins__": {}}, context):
                        # Apply action
                        await self._execute_adaptation_action(rule.action, feedback)
                        
                        rule.success_count += 1
                        rule.last_applied = datetime.now()
                        
                        logger.info(f"Applied adaptation rule: {rule.rule_id}")
                        
                except Exception as rule_error:
                    rule.failure_count += 1
                    logger.error(f"Error applying rule {rule.rule_id}: {str(rule_error)}")
                    
        except Exception as e:
            logger.error(f"Error applying adaptation rules: {str(e)}")
    
    async def _execute_adaptation_action(self, action: str, feedback: LearningFeedback):
        """Execute an adaptation action."""
        try:
            if action == "increase_model_complexity":
                # Increase model complexity for next optimization
                logger.info("Adaptation: Increasing model complexity for future optimizations")
                
            elif action == "reduce_overfitting_risk":
                # Add regularization or reduce complexity
                logger.info("Adaptation: Reducing overfitting risk for future optimizations")
                
            elif action == "optimize_training_speed":
                # Focus on faster training parameters
                logger.info("Adaptation: Optimizing for training speed")
                
            elif action == "adjust_business_focus":
                # Adjust optimization to focus on business metrics
                logger.info("Adaptation: Adjusting focus to business impact")
                
            elif action == "simplify_model_configuration":
                # Simplify model configuration to improve reliability
                logger.info("Adaptation: Simplifying model configuration")
                
        except Exception as e:
            logger.error(f"Error executing adaptation action {action}: {str(e)}")
    
    async def _store_feedback(self, feedback: LearningFeedback):
        """Store feedback in database."""
        try:
            conn = self.get_db_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO adaptive_optimization_feedback 
                (model_id, model_type, performance_metrics, hyperparameters, 
                 training_time, deployment_success, business_impact, timestamp, context)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                feedback.model_id,
                feedback.model_type,
                json.dumps(feedback.performance_metrics),
                json.dumps(feedback.hyperparameters),
                feedback.training_time,
                feedback.deployment_success,
                feedback.business_impact,
                feedback.timestamp,
                json.dumps(feedback.context)
            ))
            
            conn.commit()
            conn.close()
            
        except sqlite3.OperationalError:
            # Table might not exist, create it
            await self._create_feedback_table()
            # Retry
            await self._store_feedback(feedback)
        except Exception as e:
            logger.error(f"Error storing feedback: {str(e)}")
    
    async def _create_feedback_table(self):
        """Create feedback table if it doesn't exist."""
        try:
            conn = self.get_db_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS adaptive_optimization_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id TEXT NOT NULL,
                    model_type TEXT NOT NULL,
                    performance_metrics TEXT,
                    hyperparameters TEXT,
                    training_time REAL,
                    deployment_success BOOLEAN,
                    business_impact REAL,
                    timestamp DATETIME NOT NULL,
                    context TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error creating feedback table: {str(e)}")
    
    def get_optimization_statistics(self) -> Dict[str, Any]:
        """Get optimization statistics and insights."""
        try:
            stats = {
                'total_optimizations': len(self.optimization_history),
                'current_phase': self.current_phase.value,
                'exploration_count': self.exploration_count,
                'exploitation_count': self.exploitation_count,
                'strategy_performance': {},
                'feature_importance': self.performance_predictor.feature_importance,
                'adaptation_rules_status': []
            }
            
            # Strategy performance statistics
            for strategy, performances in self.strategy_performance.items():
                if performances:
                    stats['strategy_performance'][strategy.value] = {
                        'count': len(performances),
                        'average_performance': statistics.mean(performances),
                        'best_performance': max(performances),
                        'worst_performance': min(performances)
                    }
            
            # Adaptation rules status
            for rule in self.adaptation_rules:
                stats['adaptation_rules_status'].append({
                    'rule_id': rule.rule_id,
                    'enabled': rule.enabled,
                    'success_count': rule.success_count,
                    'failure_count': rule.failure_count,
                    'last_applied': rule.last_applied.isoformat() if rule.last_applied else None
                })
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting optimization statistics: {str(e)}")
            return {}


# Global adaptive optimizer instance
adaptive_optimizer = AdaptiveOptimizer()