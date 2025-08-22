"""
Model Validation MCP Server for Autonomous Demand Forecasting System.

This server validates new models against holdout datasets and production baselines,
ensuring only improved models are deployed to production.
"""

import asyncio
import json
import logging
import os
import sqlite3
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import statistics
import numpy as np
from scipy import stats

import mcp.server.stdio
from dotenv import load_dotenv

# MCP Server Imports
from mcp import types as mcp_types
from mcp.server.lowlevel import NotificationOptions, Server
from mcp.server.models import InitializationOptions

# Import data models
from autonomous_demand_forecasting.database.models import (
    ValidationResult, ValidationStatus, HoldoutDataset, ModelRegistry,
    AccuracyMetrics, ModelStatus
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
MIN_IMPROVEMENT_THRESHOLD = 0.03  # 3% minimum improvement required
STATISTICAL_SIGNIFICANCE_THRESHOLD = 0.05  # p-value threshold
HOLDOUT_DATASET_SIZE_RATIO = 0.2  # 20% of data for holdout
VALIDATION_WINDOW_DAYS = 30  # Days of data for validation


# --- Database Utility Functions ---
def get_db_connection():
    """Get database connection with row factory."""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def create_holdout_dataset(
    product_categories: List[str],
    date_range_start: datetime,
    date_range_end: datetime,
    sample_size: Optional[int] = None
) -> Dict[str, Any]:
    """Create a holdout dataset for model validation."""
    try:
        dataset_id = f"holdout_{uuid.uuid4().hex[:8]}_{datetime.now().strftime('%Y%m%d')}"
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # If sample_size not provided, calculate based on available data
        if sample_size is None:
            cursor.execute("""
                SELECT COUNT(*) as total_count
                FROM sales_transactions 
                WHERE transaction_date BETWEEN ? AND ?
                AND category IN ({})
            """.format(','.join('?' * len(product_categories))), 
            [date_range_start, date_range_end] + product_categories)
            
            total_count = cursor.fetchone()['total_count']
            sample_size = max(int(total_count * HOLDOUT_DATASET_SIZE_RATIO), 100)
        
        # Create holdout dataset record
        cursor.execute("""
            INSERT INTO holdout_datasets 
            (dataset_id, product_categories, date_range_start, date_range_end, 
             sample_size, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            dataset_id,
            json.dumps(product_categories),
            date_range_start,
            date_range_end,
            sample_size,
            datetime.now()
        ))
        
        conn.commit()
        conn.close()
        
        return {
            "success": True,
            "message": f"Holdout dataset created successfully",
            "dataset_id": dataset_id,
            "sample_size": sample_size,
            "categories": product_categories,
            "date_range": {
                "start": date_range_start.isoformat(),
                "end": date_range_end.isoformat()
            }
        }
    except sqlite3.Error as e:
        return {
            "success": False,
            "message": f"Error creating holdout dataset: {e}"
        }


def get_holdout_dataset(dataset_id: str) -> Optional[HoldoutDataset]:
    """Retrieve holdout dataset information."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT dataset_id, product_categories, date_range_start, date_range_end,
                   sample_size, dataset_path, created_at
            FROM holdout_datasets 
            WHERE dataset_id = ?
        """, (dataset_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return None
        
        # Parse JSON categories
        categories = json.loads(row['product_categories']) if row['product_categories'] else []
        
        # Handle datetime parsing
        created_at = row['created_at']
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        
        date_range_start = None
        date_range_end = None
        if row['date_range_start']:
            if isinstance(row['date_range_start'], str):
                date_range_start = datetime.fromisoformat(row['date_range_start'])
            else:
                date_range_start = row['date_range_start']
        
        if row['date_range_end']:
            if isinstance(row['date_range_end'], str):
                date_range_end = datetime.fromisoformat(row['date_range_end'])
            else:
                date_range_end = row['date_range_end']
        
        return HoldoutDataset(
            dataset_id=row['dataset_id'],
            product_categories=categories,
            date_range_start=date_range_start,
            date_range_end=date_range_end,
            sample_size=row['sample_size'],
            dataset_path=row['dataset_path'],
            created_at=created_at
        )
    except sqlite3.Error as e:
        logging.error(f"Error retrieving holdout dataset: {e}")
        return None


def validate_model_performance(
    model_id: str,
    holdout_dataset_id: str,
    test_accuracy: float,
    test_mape: Optional[float] = None,
    test_rmse: Optional[float] = None
) -> Dict[str, Any]:
    """Validate model performance against holdout dataset."""
    try:
        # Get holdout dataset info
        holdout_dataset = get_holdout_dataset(holdout_dataset_id)
        if not holdout_dataset:
            return {
                "success": False,
                "message": f"Holdout dataset {holdout_dataset_id} not found"
            }
        
        # Get baseline model performance for comparison
        baseline_accuracy = get_baseline_model_accuracy(model_id, holdout_dataset.product_categories)
        
        # Calculate improvement percentage
        improvement_percentage = 0.0
        if baseline_accuracy and baseline_accuracy > 0:
            improvement_percentage = (test_accuracy - baseline_accuracy) / baseline_accuracy
        
        # Determine validation status
        validation_status = ValidationStatus.FAILED
        if improvement_percentage >= MIN_IMPROVEMENT_THRESHOLD:
            validation_status = ValidationStatus.PASSED
        
        # Create validation result record
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO validation_results 
            (model_id, validation_dataset_id, validation_date, accuracy_score,
             baseline_accuracy, improvement_percentage, validation_status)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            model_id,
            holdout_dataset_id,
            datetime.now(),
            test_accuracy,
            baseline_accuracy,
            improvement_percentage,
            validation_status.value
        ))
        
        validation_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return {
            "success": True,
            "message": f"Model validation completed",
            "validation_id": validation_id,
            "validation_status": validation_status.value,
            "test_accuracy": test_accuracy,
            "baseline_accuracy": baseline_accuracy,
            "improvement_percentage": improvement_percentage,
            "passed": validation_status == ValidationStatus.PASSED
        }
    except sqlite3.Error as e:
        return {
            "success": False,
            "message": f"Error validating model performance: {e}"
        }


def get_baseline_model_accuracy(
    model_id: str,
    product_categories: List[str]
) -> Optional[float]:
    """Get baseline model accuracy for comparison."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get current production model performance for these categories
        cursor.execute("""
            SELECT AVG(accuracy_score) as avg_accuracy
            FROM model_performance mp
            JOIN model_registry mr ON mp.model_id = mr.model_id
            WHERE mr.status = 'PRODUCTION'
            AND mp.product_category IN ({})
            AND mp.timestamp >= datetime('now', '-30 days')
        """.format(','.join('?' * len(product_categories))), product_categories)
        
        result = cursor.fetchone()
        conn.close()
        
        if result and result['avg_accuracy']:
            return float(result['avg_accuracy'])
        
        # If no production model, use historical average
        return 0.75  # Default baseline
    except sqlite3.Error as e:
        logging.error(f"Error getting baseline accuracy: {e}")
        return None


def compare_models_statistical_significance(
    model_a_results: List[float],
    model_b_results: List[float],
    alpha: float = STATISTICAL_SIGNIFICANCE_THRESHOLD
) -> Dict[str, Any]:
    """
    Perform comprehensive statistical significance testing between two models.
    
    Implements multiple statistical tests including t-tests, Mann-Whitney U test,
    and Kolmogorov-Smirnov test for robust model comparison.
    """
    try:
        if len(model_a_results) < 2 or len(model_b_results) < 2:
            return {
                "success": False,
                "message": "Insufficient data for statistical testing"
            }
        
        # Convert to numpy arrays for easier computation
        a_results = np.array(model_a_results)
        b_results = np.array(model_b_results)
        
        # Basic descriptive statistics
        a_mean, a_std = np.mean(a_results), np.std(a_results, ddof=1)
        b_mean, b_std = np.mean(b_results), np.std(b_results, ddof=1)
        
        # Test for normality using Shapiro-Wilk test
        a_normal = stats.shapiro(a_results).pvalue > 0.05 if len(a_results) <= 5000 else True
        b_normal = stats.shapiro(b_results).pvalue > 0.05 if len(b_results) <= 5000 else True
        
        # Perform appropriate t-test
        if len(model_a_results) == len(model_b_results):
            # Paired t-test
            t_statistic, t_p_value = stats.ttest_rel(a_results, b_results)
            test_type = "paired_t_test"
        else:
            # Independent t-test with equal variance assumption check
            levene_stat, levene_p = stats.levene(a_results, b_results)
            equal_var = levene_p > 0.05
            
            t_statistic, t_p_value = stats.ttest_ind(a_results, b_results, equal_var=equal_var)
            test_type = f"independent_t_test_{'equal_var' if equal_var else 'unequal_var'}"
        
        # Non-parametric tests for robustness
        # Mann-Whitney U test (independent samples)
        if len(model_a_results) != len(model_b_results):
            u_statistic, u_p_value = stats.mannwhitneyu(a_results, b_results, alternative='two-sided')
        else:
            # Wilcoxon signed-rank test (paired samples)
            u_statistic, u_p_value = stats.wilcoxon(a_results, b_results, alternative='two-sided')
        
        # Kolmogorov-Smirnov test for distribution comparison
        ks_statistic, ks_p_value = stats.ks_2samp(a_results, b_results)
        
        # Calculate effect sizes
        # Cohen's d - use independent samples approach for more robust calculation
        pooled_std = np.sqrt(((len(a_results) - 1) * np.var(a_results, ddof=1) + 
                             (len(b_results) - 1) * np.var(b_results, ddof=1)) / 
                            (len(a_results) + len(b_results) - 2))
        
        # If pooled std is too small, use the larger of the two individual stds
        if pooled_std < 1e-10:
            pooled_std = max(a_std, b_std)
        
        cohens_d = (a_mean - b_mean) / pooled_std if pooled_std > 1e-10 else 0
        
        # Cap Cohen's d to reasonable range to avoid numerical issues
        cohens_d = np.clip(cohens_d, -10, 10)
        
        # Glass's delta (alternative effect size)
        glass_delta = (a_mean - b_mean) / b_std if b_std > 1e-10 else 0
        glass_delta = np.clip(glass_delta, -10, 10)  # Cap to reasonable range
        
        # Hedges' g (bias-corrected Cohen's d)
        n_total = len(a_results) + len(b_results)
        hedges_g = cohens_d * (1 - (3 / (4 * n_total - 9))) if n_total > 9 else cohens_d
        
        # Effect size interpretation
        def interpret_effect_size(d):
            abs_d = abs(d)
            if abs_d < 0.2:
                return "negligible"
            elif abs_d < 0.5:
                return "small"
            elif abs_d < 0.8:
                return "medium"
            else:
                return "large"
        
        # Determine overall significance using multiple tests
        parametric_significant = bool(t_p_value < alpha)
        nonparametric_significant = bool(u_p_value < alpha)
        distribution_different = bool(ks_p_value < alpha)
        
        # Consensus significance (majority rule)
        significance_votes = sum([parametric_significant, nonparametric_significant])
        is_significant = bool(significance_votes >= 1)  # At least one test significant
        
        # Confidence intervals
        def calculate_confidence_interval(data, confidence=0.95):
            n = len(data)
            mean = np.mean(data)
            se = stats.sem(data)
            h = se * stats.t.ppf((1 + confidence) / 2., n-1)
            return (mean - h, mean + h)
        
        a_ci = calculate_confidence_interval(a_results, 1 - alpha)
        b_ci = calculate_confidence_interval(b_results, 1 - alpha)
        
        # Power analysis (post-hoc)
        from scipy.stats import norm
        
        # Calculate observed power for t-test
        ncp = abs(cohens_d) * np.sqrt(len(a_results) * len(b_results) / (len(a_results) + len(b_results)))
        critical_t = stats.t.ppf(1 - alpha/2, len(a_results) + len(b_results) - 2)
        power = 1 - stats.t.cdf(critical_t, len(a_results) + len(b_results) - 2, ncp)
        
        return {
            "success": True,
            "test_type": test_type,  # Add for backward compatibility
            "cohens_d": float(cohens_d),  # Add for backward compatibility
            "sample_sizes": {
                "model_a": len(a_results),
                "model_b": len(b_results)
            },
            "descriptive_statistics": {
                "model_a": {
                    "mean": float(a_mean),
                    "std": float(a_std),
                    "min": float(np.min(a_results)),
                    "max": float(np.max(a_results)),
                    "median": float(np.median(a_results)),
                    "confidence_interval": [float(a_ci[0]), float(a_ci[1])],
                    "is_normal": a_normal
                },
                "model_b": {
                    "mean": float(b_mean),
                    "std": float(b_std),
                    "min": float(np.min(b_results)),
                    "max": float(np.max(b_results)),
                    "median": float(np.median(b_results)),
                    "confidence_interval": [float(b_ci[0]), float(b_ci[1])],
                    "is_normal": b_normal
                }
            },
            "parametric_tests": {
                "t_test": {
                    "type": test_type,
                    "statistic": float(t_statistic),
                    "p_value": float(t_p_value),
                    "significant": parametric_significant
                }
            },
            "nonparametric_tests": {
                "mann_whitney_u" if len(model_a_results) != len(model_b_results) else "wilcoxon": {
                    "statistic": float(u_statistic),
                    "p_value": float(u_p_value),
                    "significant": nonparametric_significant
                },
                "kolmogorov_smirnov": {
                    "statistic": float(ks_statistic),
                    "p_value": float(ks_p_value),
                    "significant": distribution_different
                }
            },
            "effect_sizes": {
                "cohens_d": {
                    "value": float(cohens_d),
                    "interpretation": interpret_effect_size(cohens_d)
                },
                "glass_delta": {
                    "value": float(glass_delta),
                    "interpretation": interpret_effect_size(glass_delta)
                },
                "hedges_g": {
                    "value": float(hedges_g),
                    "interpretation": interpret_effect_size(hedges_g)
                }
            },
            "significance_summary": {
                "alpha": alpha,
                "is_significant": is_significant,
                "parametric_significant": parametric_significant,
                "nonparametric_significant": nonparametric_significant,
                "distribution_different": distribution_different,
                "consensus": "significant" if is_significant else "not_significant"
            },
            "power_analysis": {
                "observed_power": float(power),
                "adequate_power": power >= 0.8
            },
            "practical_significance": {
                "mean_difference": float(a_mean - b_mean),
                "percent_improvement": float((a_mean - b_mean) / b_mean * 100) if b_mean != 0 else 0,
                "meets_threshold": bool(abs(a_mean - b_mean) >= MIN_IMPROVEMENT_THRESHOLD * b_mean) if b_mean != 0 else False
            }
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Error in statistical testing: {e}"
        }


def get_validation_results(
    model_id: str,
    days_back: int = VALIDATION_WINDOW_DAYS
) -> List[ValidationResult]:
    """Retrieve validation results for a model."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        cursor.execute("""
            SELECT model_id, validation_dataset_id, validation_date, accuracy_score,
                   baseline_accuracy, improvement_percentage, statistical_significance,
                   validation_status, validation_notes
            FROM validation_results 
            WHERE model_id = ? AND validation_date >= ?
            ORDER BY validation_date DESC
        """, (model_id, cutoff_date))
        
        rows = cursor.fetchall()
        conn.close()
        
        results = []
        for row in rows:
            # Handle datetime parsing
            validation_date = row['validation_date']
            if isinstance(validation_date, str):
                validation_date = datetime.fromisoformat(validation_date)
            
            result = ValidationResult(
                model_id=row['model_id'],
                validation_dataset_id=row['validation_dataset_id'],
                validation_date=validation_date,
                accuracy_score=row['accuracy_score'],
                baseline_accuracy=row['baseline_accuracy'],
                improvement_percentage=row['improvement_percentage'],
                statistical_significance=row['statistical_significance'],
                validation_status=ValidationStatus(row['validation_status']),
                validation_notes=row['validation_notes']
            )
            results.append(result)
        
        return results
    except sqlite3.Error as e:
        logging.error(f"Error retrieving validation results: {e}")
        return []


def perform_ab_testing_analysis(
    model_a_id: str,
    model_b_id: str,
    test_duration_days: int = 7,
    confidence_level: float = 0.95,
    minimum_sample_size: int = 10,
    stratify_by_category: bool = True
) -> Dict[str, Any]:
    """
    Perform comprehensive A/B testing analysis between two models.
    
    Implements advanced A/B testing methodology including:
    - Statistical power analysis
    - Multiple comparison corrections
    - Stratified analysis by product category
    - Sequential testing capabilities
    - Business impact assessment
    """
    try:
        # Get performance data for both models
        end_date = datetime.now()
        start_date = end_date - timedelta(days=test_duration_days)
        
        model_a_performance = get_model_performance_data(model_a_id, start_date, end_date)
        model_b_performance = get_model_performance_data(model_b_id, start_date, end_date)
        
        if not model_a_performance or not model_b_performance:
            return {
                "success": False,
                "message": "Insufficient performance data for A/B testing"
            }
        
        # Extract performance metrics
        model_a_data = _extract_performance_metrics(model_a_performance)
        model_b_data = _extract_performance_metrics(model_b_performance)
        
        if len(model_a_data['accuracy']) < minimum_sample_size or len(model_b_data['accuracy']) < minimum_sample_size:
            return {
                "success": False,
                "message": f"Insufficient data points for reliable A/B testing (minimum {minimum_sample_size} per model)"
            }
        
        # Overall statistical comparison
        overall_comparison = compare_models_statistical_significance(
            model_a_data['accuracy'], model_b_data['accuracy']
        )
        
        if not overall_comparison["success"]:
            return overall_comparison
        
        # Stratified analysis by product category if requested
        stratified_results = {}
        if stratify_by_category:
            categories = set(model_a_data['categories'] + model_b_data['categories'])
            
            for category in categories:
                a_cat_scores = [
                    model_a_data['accuracy'][i] for i, cat in enumerate(model_a_data['categories']) 
                    if cat == category
                ]
                b_cat_scores = [
                    model_b_data['accuracy'][i] for i, cat in enumerate(model_b_data['categories']) 
                    if cat == category
                ]
                
                if len(a_cat_scores) >= 3 and len(b_cat_scores) >= 3:  # Minimum for category analysis
                    cat_comparison = compare_models_statistical_significance(a_cat_scores, b_cat_scores)
                    if cat_comparison["success"]:
                        stratified_results[category] = {
                            "model_a_mean": cat_comparison["descriptive_statistics"]["model_a"]["mean"],
                            "model_b_mean": cat_comparison["descriptive_statistics"]["model_b"]["mean"],
                            "significant": cat_comparison["significance_summary"]["is_significant"],
                            "effect_size": cat_comparison["effect_sizes"]["cohens_d"]["value"],
                            "sample_sizes": {
                                "model_a": len(a_cat_scores),
                                "model_b": len(b_cat_scores)
                            }
                        }
        
        # Multiple comparison correction (Bonferroni)
        num_comparisons = 1 + len(stratified_results)  # Overall + category comparisons
        bonferroni_alpha = (1 - confidence_level) / num_comparisons
        
        # Sequential testing analysis (optional stopping)
        sequential_results = _perform_sequential_analysis(
            model_a_data['accuracy'], model_b_data['accuracy'], confidence_level
        )
        
        # Business impact assessment
        business_impact = _calculate_business_impact(
            model_a_data, model_b_data, model_a_id, model_b_id
        )
        
        # Power analysis and sample size recommendations
        power_analysis = _perform_power_analysis(
            model_a_data['accuracy'], model_b_data['accuracy'], confidence_level
        )
        
        # Determine winner with multiple criteria
        winner_analysis = _determine_ab_test_winner(
            overall_comparison, stratified_results, business_impact, confidence_level
        )
        
        # Generate comprehensive recommendations
        recommendations = _generate_comprehensive_ab_recommendations(
            overall_comparison, stratified_results, business_impact, 
            power_analysis, winner_analysis
        )
        
        return {
            "success": True,
            "test_metadata": {
                "model_a_id": model_a_id,
                "model_b_id": model_b_id,
                "test_period": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "duration_days": test_duration_days
                },
                "confidence_level": confidence_level,
                "bonferroni_corrected_alpha": bonferroni_alpha,
                "minimum_sample_size": minimum_sample_size
            },
            "overall_analysis": {
                "statistical_comparison": overall_comparison,
                "sample_sizes": {
                    "model_a": len(model_a_data['accuracy']),
                    "model_b": len(model_b_data['accuracy'])
                }
            },
            "stratified_analysis": {
                "enabled": stratify_by_category,
                "categories": stratified_results,
                "num_categories": len(stratified_results)
            },
            "sequential_analysis": sequential_results,
            "business_impact": business_impact,
            "power_analysis": power_analysis,
            "winner_analysis": winner_analysis,
            "recommendations": recommendations,
            "test_validity": {
                "sufficient_sample_size": len(model_a_data['accuracy']) >= minimum_sample_size and len(model_b_data['accuracy']) >= minimum_sample_size,
                "adequate_power": power_analysis.get("observed_power", 0) >= 0.8,
                "test_duration_adequate": test_duration_days >= 7,
                "data_quality_score": _assess_data_quality(model_a_performance, model_b_performance)
            }
        }
        
    except Exception as e:
        logging.error(f"Error in A/B testing analysis: {e}")
        return {
            "success": False,
            "message": f"Error in A/B testing analysis: {str(e)}"
        }


def _extract_performance_metrics(performance_data: List[Dict[str, Any]]) -> Dict[str, List]:
    """Extract and organize performance metrics from raw data."""
    return {
        'accuracy': [p['accuracy_score'] for p in performance_data if p['accuracy_score'] is not None],
        'mape': [p['mape_score'] for p in performance_data if p['mape_score'] is not None],
        'rmse': [p['rmse_score'] for p in performance_data if p['rmse_score'] is not None],
        'categories': [p['product_category'] for p in performance_data],
        'timestamps': [p['timestamp'] for p in performance_data]
    }


def _perform_sequential_analysis(
    model_a_results: List[float], 
    model_b_results: List[float], 
    confidence_level: float
) -> Dict[str, Any]:
    """
    Perform sequential analysis to determine if test can be stopped early.
    
    Implements Sequential Probability Ratio Test (SPRT) methodology.
    """
    try:
        from scipy.stats import norm
        
        alpha = 1 - confidence_level
        beta = 0.2  # 80% power
        
        # Calculate sequential boundaries
        a = np.log(beta / (1 - alpha))  # Lower boundary
        b = np.log((1 - beta) / alpha)  # Upper boundary
        
        # Calculate cumulative log likelihood ratios
        n_min = min(len(model_a_results), len(model_b_results))
        cumulative_llr = []
        
        for i in range(1, n_min + 1):
            a_sample = model_a_results[:i]
            b_sample = model_b_results[:i]
            
            # Simple difference in means as test statistic
            diff = np.mean(a_sample) - np.mean(b_sample)
            se = np.sqrt(np.var(a_sample, ddof=1)/len(a_sample) + np.var(b_sample, ddof=1)/len(b_sample))
            
            if se > 0:
                z_score = diff / se
                llr = z_score  # Simplified LLR
                cumulative_llr.append(llr)
            else:
                cumulative_llr.append(0)
        
        # Determine if boundaries are crossed
        early_stop_point = None
        decision = "continue"
        
        for i, llr in enumerate(cumulative_llr):
            if llr >= b:
                early_stop_point = i + 1
                decision = "model_a_wins"
                break
            elif llr <= a:
                early_stop_point = i + 1
                decision = "model_b_wins"
                break
        
        return {
            "can_stop_early": early_stop_point is not None,
            "early_stop_point": early_stop_point,
            "decision": decision,
            "boundaries": {"lower": float(a), "upper": float(b)},
            "cumulative_llr": [float(x) for x in cumulative_llr],
            "recommendation": "Stop test early" if early_stop_point else "Continue test"
        }
        
    except Exception as e:
        return {
            "error": f"Sequential analysis failed: {e}",
            "can_stop_early": False
        }


def _calculate_business_impact(
    model_a_data: Dict[str, List], 
    model_b_data: Dict[str, List],
    model_a_id: str,
    model_b_id: str
) -> Dict[str, Any]:
    """Calculate business impact metrics for A/B test results."""
    try:
        a_mean_accuracy = np.mean(model_a_data['accuracy'])
        b_mean_accuracy = np.mean(model_b_data['accuracy'])
        
        # Calculate improvement metrics
        absolute_improvement = a_mean_accuracy - b_mean_accuracy
        relative_improvement = (absolute_improvement / b_mean_accuracy) * 100 if b_mean_accuracy > 0 else 0
        
        # Estimate business value (simplified model)
        # Assume 1% accuracy improvement = $10,000 monthly value
        estimated_monthly_value = absolute_improvement * 100 * 10000
        estimated_annual_value = estimated_monthly_value * 12
        
        # Risk assessment
        a_std = np.std(model_a_data['accuracy'])
        b_std = np.std(model_b_data['accuracy'])
        
        risk_score = abs(a_std - b_std) / max(a_std, b_std) if max(a_std, b_std) > 0 else 0
        risk_level = "high" if risk_score > 0.5 else "medium" if risk_score > 0.2 else "low"
        
        return {
            "accuracy_improvement": {
                "absolute": float(absolute_improvement),
                "relative_percent": float(relative_improvement),
                "meets_business_threshold": abs(relative_improvement) >= 3.0  # 3% threshold
            },
            "estimated_value": {
                "monthly_usd": float(estimated_monthly_value),
                "annual_usd": float(estimated_annual_value),
                "confidence": "medium"  # Based on simplified model
            },
            "risk_assessment": {
                "variability_risk": risk_level,
                "risk_score": float(risk_score),
                "recommendation": "low_risk" if risk_level == "low" else "monitor_closely"
            },
            "winner": model_a_id if absolute_improvement > 0 else model_b_id,
            "confidence_in_winner": "high" if abs(relative_improvement) > 5 else "medium" if abs(relative_improvement) > 2 else "low"
        }
        
    except Exception as e:
        return {
            "error": f"Business impact calculation failed: {e}",
            "accuracy_improvement": {"absolute": 0, "relative_percent": 0}
        }


def _perform_power_analysis(
    model_a_results: List[float], 
    model_b_results: List[float], 
    confidence_level: float
) -> Dict[str, Any]:
    """Perform statistical power analysis and sample size recommendations."""
    try:
        from scipy.stats import norm, t
        
        alpha = 1 - confidence_level
        
        # Calculate observed effect size
        a_mean, a_std = np.mean(model_a_results), np.std(model_a_results, ddof=1)
        b_mean, b_std = np.mean(model_b_results), np.std(model_b_results, ddof=1)
        
        pooled_std = np.sqrt((a_std**2 + b_std**2) / 2)
        observed_effect_size = abs(a_mean - b_mean) / pooled_std if pooled_std > 0 else 0
        
        # Calculate observed power
        n_a, n_b = len(model_a_results), len(model_b_results)
        ncp = observed_effect_size * np.sqrt(n_a * n_b / (n_a + n_b))
        df = n_a + n_b - 2
        critical_t = t.ppf(1 - alpha/2, df)
        observed_power = 1 - t.cdf(critical_t, df, ncp) + t.cdf(-critical_t, df, ncp)
        
        # Sample size recommendations for different effect sizes
        effect_sizes = [0.2, 0.5, 0.8]  # Small, medium, large
        sample_size_recommendations = {}
        
        for effect_size in effect_sizes:
            # Cohen's formula for sample size
            z_alpha = norm.ppf(1 - alpha/2)
            z_beta = norm.ppf(0.8)  # 80% power
            n_per_group = 2 * ((z_alpha + z_beta) / effect_size)**2
            
            sample_size_recommendations[f"effect_size_{effect_size}"] = {
                "n_per_group": int(np.ceil(n_per_group)),
                "total_n": int(np.ceil(n_per_group * 2)),
                "description": f"Sample size needed to detect effect size {effect_size} with 80% power"
            }
        
        return {
            "observed_effect_size": float(observed_effect_size),
            "observed_power": float(observed_power),
            "adequate_power": observed_power >= 0.8,
            "current_sample_sizes": {"model_a": n_a, "model_b": n_b},
            "sample_size_recommendations": sample_size_recommendations,
            "power_interpretation": {
                "level": "high" if observed_power >= 0.8 else "medium" if observed_power >= 0.6 else "low",
                "recommendation": "Sufficient power" if observed_power >= 0.8 else "Consider increasing sample size"
            }
        }
        
    except Exception as e:
        return {
            "error": f"Power analysis failed: {e}",
            "observed_power": 0,
            "adequate_power": False
        }


def _determine_ab_test_winner(
    overall_comparison: Dict[str, Any],
    stratified_results: Dict[str, Any],
    business_impact: Dict[str, Any],
    confidence_level: float
) -> Dict[str, Any]:
    """Determine A/B test winner using multiple criteria."""
    try:
        # Statistical significance
        statistically_significant = overall_comparison["significance_summary"]["is_significant"]
        
        # Practical significance
        practically_significant = business_impact["accuracy_improvement"]["meets_business_threshold"]
        
        # Consistency across categories
        category_consistency = 0
        if stratified_results:
            consistent_categories = sum(
                1 for cat_result in stratified_results.values()
                if cat_result["significant"] and 
                   np.sign(cat_result["model_a_mean"] - cat_result["model_b_mean"]) == 
                   np.sign(overall_comparison["descriptive_statistics"]["model_a"]["mean"] - 
                          overall_comparison["descriptive_statistics"]["model_b"]["mean"])
            )
            category_consistency = consistent_categories / len(stratified_results) if stratified_results else 0
        
        # Overall confidence score
        confidence_factors = [
            statistically_significant,
            practically_significant,
            category_consistency >= 0.7,  # At least 70% of categories consistent
            overall_comparison["power_analysis"]["adequate_power"]
        ]
        
        confidence_score = sum(confidence_factors) / len(confidence_factors)
        
        # Determine winner
        a_mean = overall_comparison["descriptive_statistics"]["model_a"]["mean"]
        b_mean = overall_comparison["descriptive_statistics"]["model_b"]["mean"]
        
        if a_mean > b_mean:
            winner = "model_a"
            winner_confidence = confidence_score
        elif b_mean > a_mean:
            winner = "model_b"
            winner_confidence = confidence_score
        else:
            winner = "tie"
            winner_confidence = 0
        
        return {
            "winner": winner,
            "confidence_score": float(confidence_score),
            "confidence_level": "high" if confidence_score >= 0.8 else "medium" if confidence_score >= 0.6 else "low",
            "criteria_met": {
                "statistically_significant": statistically_significant,
                "practically_significant": practically_significant,
                "category_consistent": category_consistency >= 0.7,
                "adequate_power": overall_comparison.get("power_analysis", {}).get("adequate_power", False)
            },
            "category_consistency_score": float(category_consistency),
            "recommendation": _get_winner_recommendation(winner, confidence_score, statistically_significant, practically_significant)
        }
        
    except Exception as e:
        return {
            "error": f"Winner determination failed: {e}",
            "winner": "undetermined",
            "confidence_score": 0
        }


def _get_winner_recommendation(
    winner: str, 
    confidence_score: float, 
    statistically_significant: bool, 
    practically_significant: bool
) -> str:
    """Generate recommendation based on winner analysis."""
    if winner == "tie":
        return "No significant difference detected - continue with current model"
    
    if confidence_score >= 0.8:
        return f"Strong evidence for {winner} - recommend deployment"
    elif confidence_score >= 0.6:
        if statistically_significant and practically_significant:
            return f"Moderate evidence for {winner} - recommend deployment with monitoring"
        else:
            return f"Weak evidence for {winner} - consider extending test or gathering more data"
    else:
        return "Insufficient evidence for decision - extend test duration or increase sample size"


def _generate_comprehensive_ab_recommendations(
    overall_comparison: Dict[str, Any],
    stratified_results: Dict[str, Any],
    business_impact: Dict[str, Any],
    power_analysis: Dict[str, Any],
    winner_analysis: Dict[str, Any]
) -> List[str]:
    """Generate comprehensive recommendations based on A/B test results."""
    recommendations = []
    
    # Statistical recommendations
    if not overall_comparison["significance_summary"]["is_significant"]:
        recommendations.append("No statistically significant difference detected - consider longer test duration")
    
    if not power_analysis.get("adequate_power", False):
        recommendations.append("Test power is below 80% - consider increasing sample size for more reliable results")
    
    # Business impact recommendations
    if not business_impact["accuracy_improvement"]["meets_business_threshold"]:
        recommendations.append("Improvement does not meet business significance threshold (3%) - evaluate if change is worthwhile")
    
    # Category-specific recommendations
    if stratified_results:
        inconsistent_categories = [
            cat for cat, result in stratified_results.items()
            if not result["significant"] or result.get("effect_size", 0) < 0.2
        ]
        if inconsistent_categories:
            recommendations.append(f"Inconsistent results in categories: {', '.join(inconsistent_categories)} - investigate category-specific factors")
    
    # Winner-based recommendations
    if winner_analysis["confidence_score"] >= 0.8:
        recommendations.append(f"High confidence in {winner_analysis['winner']} - proceed with deployment")
    elif winner_analysis["confidence_score"] >= 0.6:
        recommendations.append(f"Moderate confidence in {winner_analysis['winner']} - deploy with enhanced monitoring")
    else:
        recommendations.append("Low confidence in results - extend test or investigate data quality issues")
    
    # Risk-based recommendations
    risk_level = business_impact.get("risk_assessment", {}).get("variability_risk", "medium")
    if risk_level == "high":
        recommendations.append("High variability detected - implement gradual rollout strategy")
    
    return recommendations


def _assess_data_quality(
    model_a_performance: List[Dict[str, Any]], 
    model_b_performance: List[Dict[str, Any]]
) -> float:
    """Assess data quality for A/B test validity."""
    try:
        # Check for missing values
        a_missing = sum(1 for p in model_a_performance if p['accuracy_score'] is None)
        b_missing = sum(1 for p in model_b_performance if p['accuracy_score'] is None)
        
        total_a = len(model_a_performance)
        total_b = len(model_b_performance)
        
        completeness_a = (total_a - a_missing) / total_a if total_a > 0 else 0
        completeness_b = (total_b - b_missing) / total_b if total_b > 0 else 0
        
        # Check for temporal coverage
        if model_a_performance and model_b_performance:
            a_timestamps = [p['timestamp'] for p in model_a_performance if p['timestamp']]
            b_timestamps = [p['timestamp'] for p in model_b_performance if p['timestamp']]
            
            if a_timestamps and b_timestamps:
                a_span = (max(a_timestamps) - min(a_timestamps)).total_seconds()
                b_span = (max(b_timestamps) - min(b_timestamps)).total_seconds()
                temporal_balance = min(a_span, b_span) / max(a_span, b_span) if max(a_span, b_span) > 0 else 1
            else:
                temporal_balance = 0
        else:
            temporal_balance = 0
        
        # Overall quality score
        quality_score = (completeness_a + completeness_b + temporal_balance) / 3
        
        return float(quality_score)
        
    except Exception:
        return 0.5  # Default moderate quality score


# --- Business Impact Assessment Functions ---

def calculate_business_impact(
    model_id: str,
    deployment_id: str,
    baseline_accuracy: float,
    improved_accuracy: float,
    product_categories: List[str],
    time_period_days: int = 30
) -> Dict[str, Any]:
    """
    Calculate comprehensive business impact of model improvements.
    
    Implements advanced business impact assessment including:
    - Revenue impact calculation based on accuracy improvements
    - Inventory optimization benefits
    - Cost reduction from better demand forecasting
    - Risk mitigation value assessment
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get historical sales data for impact calculation
        end_date = datetime.now()
        start_date = end_date - timedelta(days=time_period_days)
        
        # Calculate total sales volume for affected categories
        cursor.execute("""
            SELECT 
                category,
                SUM(total_amount) as total_revenue,
                SUM(quantity) as total_quantity,
                COUNT(*) as transaction_count,
                AVG(unit_price) as avg_price
            FROM sales_transactions 
            WHERE category IN ({}) 
            AND transaction_date BETWEEN ? AND ?
            GROUP BY category
        """.format(','.join('?' * len(product_categories))), 
        product_categories + [start_date, end_date])
        
        sales_data = cursor.fetchall()
        
        if not sales_data:
            return {
                "success": False,
                "message": "No sales data available for business impact calculation"
            }
        
        # Calculate category-specific impacts
        category_impacts = {}
        total_revenue_impact = 0
        total_cost_savings = 0
        
        for row in sales_data:
            category = row['category']
            revenue = row['total_revenue']
            quantity = row['total_quantity']
            avg_price = row['avg_price']
            
            # Calculate accuracy improvement impact
            accuracy_improvement = improved_accuracy - baseline_accuracy
            
            # Revenue impact calculation
            # Better forecasting reduces stockouts and overstock situations
            # Assume 1% accuracy improvement = 0.5% revenue improvement (conservative estimate)
            revenue_improvement_rate = accuracy_improvement * 0.5
            category_revenue_impact = revenue * revenue_improvement_rate
            
            # Cost savings from better inventory management
            # Reduced holding costs and stockout costs
            inventory_cost_rate = 0.15  # 15% of revenue as inventory carrying cost
            cost_reduction_rate = accuracy_improvement * 0.3  # 30% of accuracy improvement
            category_cost_savings = revenue * inventory_cost_rate * cost_reduction_rate
            
            # Stockout reduction impact
            stockout_impact = _calculate_stockout_reduction_impact(
                category, accuracy_improvement, quantity, avg_price, time_period_days
            )
            
            category_impacts[category] = {
                "baseline_revenue": float(revenue),
                "revenue_impact": float(category_revenue_impact),
                "cost_savings": float(category_cost_savings),
                "stockout_reduction": stockout_impact,
                "total_impact": float(category_revenue_impact + category_cost_savings + stockout_impact),
                "accuracy_improvement": float(accuracy_improvement),
                "transaction_count": row['transaction_count']
            }
            
            total_revenue_impact += category_revenue_impact
            total_cost_savings += category_cost_savings
        
        # Calculate overall business metrics
        total_baseline_revenue = sum(row['total_revenue'] for row in sales_data)
        total_business_impact = total_revenue_impact + total_cost_savings
        roi_percentage = (total_business_impact / total_baseline_revenue) * 100 if total_baseline_revenue > 0 else 0
        
        # Risk assessment and confidence intervals
        risk_assessment = _assess_business_impact_risk(
            accuracy_improvement, category_impacts, total_baseline_revenue
        )
        
        # Store business impact in database
        impact_metrics = [
            ("revenue_impact", total_baseline_revenue, total_baseline_revenue + total_revenue_impact, 
             (total_revenue_impact / total_baseline_revenue) * 100 if total_baseline_revenue > 0 else 0, total_revenue_impact),
            ("cost_savings", 0, total_cost_savings, 0, total_cost_savings),
            ("total_business_impact", total_baseline_revenue, total_baseline_revenue + total_business_impact,
             roi_percentage, total_business_impact)
        ]
        
        for metric_type, baseline, improved, improvement_pct, revenue_impact in impact_metrics:
            cursor.execute("""
                INSERT INTO business_impact 
                (deployment_id, metric_type, baseline_value, improved_value, 
                 improvement_percentage, revenue_impact, calculated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (deployment_id, metric_type, baseline, improved, improvement_pct, revenue_impact, datetime.now()))
        
        conn.commit()
        conn.close()
        
        return {
            "success": True,
            "model_id": model_id,
            "deployment_id": deployment_id,
            "calculation_period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "days": time_period_days
            },
            "accuracy_metrics": {
                "baseline_accuracy": float(baseline_accuracy),
                "improved_accuracy": float(improved_accuracy),
                "improvement": float(improved_accuracy - baseline_accuracy),
                "improvement_percentage": float((improved_accuracy - baseline_accuracy) / baseline_accuracy * 100) if baseline_accuracy > 0 else 0
            },
            "financial_impact": {
                "total_baseline_revenue": float(total_baseline_revenue),
                "revenue_impact": float(total_revenue_impact),
                "cost_savings": float(total_cost_savings),
                "total_business_impact": float(total_business_impact),
                "roi_percentage": float(roi_percentage)
            },
            "category_breakdown": category_impacts,
            "risk_assessment": risk_assessment,
            "confidence_level": "high" if risk_assessment.get("confidence_score", 0) > 0.8 else "medium" if risk_assessment.get("confidence_score", 0) > 0.6 else "low",
            "recommendation": _generate_business_impact_recommendation(total_business_impact, roi_percentage, risk_assessment)
        }
        
    except sqlite3.Error as e:
        return {
            "success": False,
            "message": f"Database error in business impact calculation: {e}"
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Error calculating business impact: {e}"
        }


def _calculate_stockout_reduction_impact(
    category: str,
    accuracy_improvement: float,
    total_quantity: int,
    avg_price: float,
    time_period_days: int
) -> float:
    """Calculate the business impact of reduced stockouts due to better forecasting."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get historical stockout data for this category
        end_date = datetime.now()
        start_date = end_date - timedelta(days=time_period_days)
        
        cursor.execute("""
            SELECT 
                COUNT(*) as stockout_count,
                AVG(duration_hours) as avg_duration,
                SUM(lost_sales_estimate) as total_lost_sales
            FROM stockout_events se
            JOIN products p ON se.product_id = p.id
            WHERE p.category = ? 
            AND se.stockout_date BETWEEN ? AND ?
        """, (category, start_date, end_date))
        
        stockout_data = cursor.fetchone()
        conn.close()
        
        if not stockout_data or stockout_data['stockout_count'] == 0:
            return 0.0
        
        # Calculate stockout reduction based on accuracy improvement
        # Assume better forecasting reduces stockout frequency by accuracy_improvement * 50%
        stockout_reduction_rate = accuracy_improvement * 0.5
        
        # Estimate lost sales impact
        if stockout_data['total_lost_sales']:
            lost_sales_reduction = stockout_data['total_lost_sales'] * stockout_reduction_rate
        else:
            # Estimate lost sales if not recorded
            # Assume each stockout loses 10% of normal sales for that period
            estimated_lost_sales = total_quantity * 0.1 * avg_price
            lost_sales_reduction = estimated_lost_sales * stockout_reduction_rate
        
        return float(lost_sales_reduction)
        
    except Exception as e:
        logging.error(f"Error calculating stockout reduction impact: {e}")
        return 0.0


def _assess_business_impact_risk(
    accuracy_improvement: float,
    category_impacts: Dict[str, Any],
    total_baseline_revenue: float
) -> Dict[str, Any]:
    """Assess risk factors in business impact calculations."""
    try:
        # Risk factors assessment
        risk_factors = []
        
        # Accuracy improvement risk
        if accuracy_improvement < 0.02:  # Less than 2% improvement
            risk_factors.append("low_accuracy_improvement")
        elif accuracy_improvement > 0.15:  # More than 15% improvement (potentially unrealistic)
            risk_factors.append("high_accuracy_improvement")
        
        # Revenue concentration risk
        if category_impacts:
            category_revenues = [impact["baseline_revenue"] for impact in category_impacts.values()]
            max_category_revenue = max(category_revenues)
            concentration_ratio = max_category_revenue / total_baseline_revenue if total_baseline_revenue > 0 else 0
            
            if concentration_ratio > 0.7:  # One category dominates 70%+ of revenue
                risk_factors.append("high_revenue_concentration")
        
        # Sample size risk
        total_transactions = sum(impact.get("transaction_count", 0) for impact in category_impacts.values())
        if total_transactions < 100:
            risk_factors.append("low_sample_size")
        
        # Calculate confidence score
        confidence_score = 1.0
        for risk_factor in risk_factors:
            if risk_factor in ["low_accuracy_improvement", "high_accuracy_improvement"]:
                confidence_score *= 0.8
            elif risk_factor == "high_revenue_concentration":
                confidence_score *= 0.9
            elif risk_factor == "low_sample_size":
                confidence_score *= 0.7
        
        # Risk level classification
        if confidence_score > 0.8:
            risk_level = "low"
        elif confidence_score > 0.6:
            risk_level = "medium"
        else:
            risk_level = "high"
        
        return {
            "risk_factors": risk_factors,
            "confidence_score": float(confidence_score),
            "risk_level": risk_level,
            "total_transactions": total_transactions,
            "revenue_concentration": float(concentration_ratio) if 'concentration_ratio' in locals() else 0,
            "mitigation_recommendations": _generate_risk_mitigation_recommendations(risk_factors, risk_level)
        }
        
    except Exception as e:
        return {
            "error": f"Risk assessment failed: {e}",
            "confidence_score": 0.5,
            "risk_level": "medium"
        }


def _generate_business_impact_recommendation(
    total_business_impact: float,
    roi_percentage: float,
    risk_assessment: Dict[str, Any]
) -> str:
    """Generate business impact-based recommendation."""
    risk_level = risk_assessment.get("risk_level", "medium")
    
    if total_business_impact > 10000 and roi_percentage > 2.0 and risk_level == "low":
        return "Strong business case for deployment - high impact with low risk"
    elif total_business_impact > 5000 and roi_percentage > 1.0 and risk_level in ["low", "medium"]:
        return "Positive business case for deployment - moderate impact with acceptable risk"
    elif total_business_impact > 1000 and roi_percentage > 0.5:
        return "Marginal business case - deploy with enhanced monitoring"
    elif risk_level == "high":
        return "High risk deployment - require additional validation before proceeding"
    else:
        return "Insufficient business impact - consider alternative improvements"


def create_approval_workflow(
    model_id: str,
    validation_result: ValidationResult,
    business_impact: Dict[str, Any],
    approval_thresholds: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Create automated approval workflow based on validation results and business impact.
    
    Implements multi-criteria approval decision making including:
    - Statistical significance requirements
    - Business impact thresholds
    - Risk assessment criteria
    - Automated escalation procedures
    """
    try:
        # Default approval thresholds
        if approval_thresholds is None:
            approval_thresholds = {
                "min_accuracy_improvement": MIN_IMPROVEMENT_THRESHOLD,  # 3%
                "min_statistical_significance": 0.05,  # p-value < 0.05
                "min_business_impact": 1000,  # $1000 minimum impact
                "min_roi_percentage": 0.5,  # 0.5% ROI minimum
                "max_risk_level": "medium"  # Accept low and medium risk
            }
        
        # Evaluation criteria
        criteria_results = {}
        
        # Statistical validation criteria
        criteria_results["statistical_significance"] = {
            "passed": validation_result.statistical_significance is not None and validation_result.statistical_significance < approval_thresholds["min_statistical_significance"],
            "value": validation_result.statistical_significance,
            "threshold": approval_thresholds["min_statistical_significance"],
            "weight": 0.3
        }
        
        criteria_results["accuracy_improvement"] = {
            "passed": validation_result.improvement_percentage is not None and validation_result.improvement_percentage >= approval_thresholds["min_accuracy_improvement"],
            "value": validation_result.improvement_percentage,
            "threshold": approval_thresholds["min_accuracy_improvement"],
            "weight": 0.25
        }
        
        # Business impact criteria
        if business_impact.get("success", False):
            financial_impact = business_impact.get("financial_impact", {})
            
            criteria_results["business_impact"] = {
                "passed": financial_impact.get("total_business_impact", 0) >= approval_thresholds["min_business_impact"],
                "value": financial_impact.get("total_business_impact", 0),
                "threshold": approval_thresholds["min_business_impact"],
                "weight": 0.25
            }
            
            criteria_results["roi_percentage"] = {
                "passed": financial_impact.get("roi_percentage", 0) >= approval_thresholds["min_roi_percentage"],
                "value": financial_impact.get("roi_percentage", 0),
                "threshold": approval_thresholds["min_roi_percentage"],
                "weight": 0.1
            }
            
            # Risk assessment criteria
            risk_assessment = business_impact.get("risk_assessment", {})
            risk_level = risk_assessment.get("risk_level", "high")
            
            risk_levels = {"low": 1, "medium": 2, "high": 3}
            max_risk_numeric = risk_levels.get(approval_thresholds["max_risk_level"], 2)
            current_risk_numeric = risk_levels.get(risk_level, 3)
            
            criteria_results["risk_assessment"] = {
                "passed": current_risk_numeric <= max_risk_numeric,
                "value": risk_level,
                "threshold": approval_thresholds["max_risk_level"],
                "weight": 0.1
            }
        else:
            # If business impact calculation failed, use conservative approach
            criteria_results["business_impact"] = {
                "passed": False,
                "value": 0,
                "threshold": approval_thresholds["min_business_impact"],
                "weight": 0.25,
                "error": "Business impact calculation failed"
            }
            
            criteria_results["roi_percentage"] = {
                "passed": False,
                "value": 0,
                "threshold": approval_thresholds["min_roi_percentage"],
                "weight": 0.1,
                "error": "ROI calculation failed"
            }
            
            criteria_results["risk_assessment"] = {
                "passed": False,
                "value": "high",
                "threshold": approval_thresholds["max_risk_level"],
                "weight": 0.1,
                "error": "Risk assessment failed"
            }
        
        # Calculate weighted approval score
        total_weight = sum(criteria["weight"] for criteria in criteria_results.values())
        weighted_score = sum(
            criteria["weight"] * (1 if criteria["passed"] else 0) 
            for criteria in criteria_results.values()
        ) / total_weight if total_weight > 0 else 0
        
        # Determine approval decision
        approval_decision = _determine_approval_decision(criteria_results, weighted_score)
        
        # Generate approval workflow record
        workflow_id = f"approval_{model_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Store approval workflow in database
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Create approval workflow record (using validation_results table with special notes)
        cursor.execute("""
            INSERT INTO validation_results 
            (model_id, validation_dataset_id, validation_date, accuracy_score,
             baseline_accuracy, improvement_percentage, statistical_significance,
             validation_status, validation_notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            model_id,
            f"approval_workflow_{workflow_id}",
            datetime.now(),
            validation_result.accuracy_score,
            validation_result.baseline_accuracy,
            validation_result.improvement_percentage,
            validation_result.statistical_significance,
            "PASSED" if approval_decision["approved"] else "FAILED",
            json.dumps({
                "workflow_type": "approval",
                "workflow_id": workflow_id,
                "criteria_results": criteria_results,
                "approval_decision": approval_decision,
                "business_impact_summary": business_impact.get("financial_impact", {}),
                "thresholds_used": approval_thresholds
            })
        ))
        
        conn.commit()
        conn.close()
        
        return {
            "success": True,
            "workflow_id": workflow_id,
            "model_id": model_id,
            "approval_decision": approval_decision,
            "criteria_evaluation": criteria_results,
            "weighted_score": float(weighted_score),
            "thresholds_used": approval_thresholds,
            "created_at": datetime.now().isoformat(),
            "next_steps": _generate_approval_next_steps(approval_decision, criteria_results)
        }
        
    except sqlite3.Error as e:
        return {
            "success": False,
            "message": f"Database error in approval workflow: {e}"
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Error creating approval workflow: {e}"
        }


def _determine_approval_decision(
    criteria_results: Dict[str, Any],
    weighted_score: float
) -> Dict[str, Any]:
    """Determine approval decision based on criteria evaluation."""
    # Critical criteria that must pass
    critical_criteria = ["statistical_significance", "accuracy_improvement"]
    critical_failures = [
        criterion for criterion in critical_criteria 
        if not criteria_results.get(criterion, {}).get("passed", False)
    ]
    
    # Business criteria
    business_criteria = ["business_impact", "roi_percentage", "risk_assessment"]
    business_failures = [
        criterion for criterion in business_criteria 
        if not criteria_results.get(criterion, {}).get("passed", False)
    ]
    
    # Decision logic
    if critical_failures:
        decision = "rejected"
        reason = f"Critical criteria failed: {', '.join(critical_failures)}"
        confidence = "high"
    elif len(business_failures) >= 2:
        decision = "rejected"
        reason = f"Multiple business criteria failed: {', '.join(business_failures)}"
        confidence = "high"
    elif weighted_score >= 0.8:
        decision = "approved"
        reason = "All criteria met with high confidence"
        confidence = "high"
    elif weighted_score >= 0.6:
        decision = "approved_conditional"
        reason = "Most criteria met - approve with enhanced monitoring"
        confidence = "medium"
    elif weighted_score >= 0.4:
        decision = "escalation_required"
        reason = "Mixed results - requires manual review"
        confidence = "low"
    else:
        decision = "rejected"
        reason = "Insufficient criteria met"
        confidence = "high"
    
    return {
        "approved": decision in ["approved", "approved_conditional"],
        "decision": decision,
        "reason": reason,
        "confidence": confidence,
        "weighted_score": float(weighted_score),
        "critical_failures": critical_failures,
        "business_failures": business_failures,
        "requires_escalation": decision == "escalation_required"
    }


def _generate_approval_next_steps(
    approval_decision: Dict[str, Any],
    criteria_results: Dict[str, Any]
) -> List[str]:
    """Generate next steps based on approval decision."""
    next_steps = []
    
    decision = approval_decision["decision"]
    
    if decision == "approved":
        next_steps.extend([
            "Proceed with model deployment using blue-green strategy",
            "Implement production monitoring with enhanced alerting",
            "Schedule post-deployment business impact validation",
            "Document deployment for audit trail"
        ])
    elif decision == "approved_conditional":
        next_steps.extend([
            "Deploy with enhanced monitoring and gradual rollout",
            "Implement additional performance safeguards",
            "Schedule accelerated review cycle (weekly vs monthly)",
            "Prepare rollback procedures with automated triggers"
        ])
    elif decision == "escalation_required":
        next_steps.extend([
            "Escalate to data science team for manual review",
            "Provide detailed analysis report to stakeholders",
            "Consider extended validation period with additional data",
            "Review approval thresholds and criteria weights"
        ])
    else:  # rejected
        next_steps.extend([
            "Do not deploy - retain current production model",
            "Investigate root causes of validation failures",
            "Consider alternative model architectures or training approaches",
            "Schedule retraining with enhanced data or features"
        ])
    
    # Add specific recommendations based on failed criteria
    failed_criteria = [
        criterion for criterion, result in criteria_results.items()
        if not result.get("passed", False)
    ]
    
    for criterion in failed_criteria:
        if criterion == "statistical_significance":
            next_steps.append("Collect additional validation data to improve statistical power")
        elif criterion == "accuracy_improvement":
            next_steps.append("Investigate feature engineering or model architecture improvements")
        elif criterion == "business_impact":
            next_steps.append("Review business impact calculation methodology and assumptions")
        elif criterion == "risk_assessment":
            next_steps.append("Implement additional risk mitigation measures before deployment")
    
    return next_steps


def track_validation_audit_trail(
    model_id: str,
    validation_id: int,
    action: str,
    user_id: Optional[str] = None,
    notes: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Track validation and approval audit trail for compliance and monitoring.
    
    Maintains comprehensive audit log including:
    - All validation actions and decisions
    - User interactions and overrides
    - System-generated approvals and rejections
    - Business impact assessments
    - Deployment authorizations
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Create audit trail entry using validation_results table with special structure
        audit_id = f"audit_{validation_id}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        audit_record = {
            "audit_id": audit_id,
            "model_id": model_id,
            "validation_id": validation_id,
            "action": action,
            "user_id": user_id,
            "timestamp": datetime.now().isoformat(),
            "notes": notes,
            "metadata": metadata or {}
        }
        
        cursor.execute("""
            INSERT INTO validation_results 
            (model_id, validation_dataset_id, validation_date, validation_status, validation_notes)
            VALUES (?, ?, ?, ?, ?)
        """, (
            model_id,
            f"audit_trail_{audit_id}",
            datetime.now(),
            "PENDING",  # Audit entries use PENDING status
            json.dumps(audit_record)
        ))
        
        audit_record_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return {
            "success": True,
            "audit_id": audit_id,
            "audit_record_id": audit_record_id,
            "model_id": model_id,
            "validation_id": validation_id,
            "action": action,
            "timestamp": audit_record["timestamp"],
            "message": f"Audit trail entry created for {action}"
        }
        
    except sqlite3.Error as e:
        return {
            "success": False,
            "message": f"Database error in audit trail tracking: {e}"
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Error tracking audit trail: {e}"
        }


def get_validation_audit_trail(
    model_id: Optional[str] = None,
    validation_id: Optional[int] = None,
    days_back: int = 30,
    action_filter: Optional[str] = None
) -> Dict[str, Any]:
    """Retrieve validation audit trail records."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Build query conditions
        conditions = ["validation_dataset_id LIKE 'audit_trail_%'"]
        params = []
        
        if model_id:
            conditions.append("model_id = ?")
            params.append(model_id)
        
        if validation_id:
            conditions.append("validation_notes LIKE ?")
            params.append(f'%"validation_id": {validation_id}%')
        
        if days_back:
            cutoff_date = datetime.now() - timedelta(days=days_back)
            conditions.append("validation_date >= ?")
            params.append(cutoff_date)
        
        query = f"""
            SELECT id, model_id, validation_dataset_id, validation_date, validation_notes
            FROM validation_results 
            WHERE {' AND '.join(conditions)}
            ORDER BY validation_date DESC
        """
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        # Parse audit records
        audit_records = []
        for row in rows:
            try:
                audit_data = json.loads(row['validation_notes'])
                
                # Filter by action if specified
                if action_filter and audit_data.get("action") != action_filter:
                    continue
                
                audit_records.append({
                    "record_id": row['id'],
                    "audit_id": audit_data.get("audit_id"),
                    "model_id": audit_data.get("model_id"),
                    "validation_id": audit_data.get("validation_id"),
                    "action": audit_data.get("action"),
                    "user_id": audit_data.get("user_id"),
                    "timestamp": audit_data.get("timestamp"),
                    "notes": audit_data.get("notes"),
                    "metadata": audit_data.get("metadata", {})
                })
            except (json.JSONDecodeError, KeyError) as e:
                logging.warning(f"Failed to parse audit record {row['id']}: {e}")
                continue
        
        return {
            "success": True,
            "audit_records": audit_records,
            "total_records": len(audit_records),
            "query_parameters": {
                "model_id": model_id,
                "validation_id": validation_id,
                "days_back": days_back,
                "action_filter": action_filter
            }
        }
        
    except sqlite3.Error as e:
        return {
            "success": False,
            "message": f"Database error retrieving audit trail: {e}"
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Error retrieving audit trail: {e}"
        }


def get_model_performance_data(
    model_id: str,
    start_date: datetime,
    end_date: datetime
) -> List[Dict[str, Any]]:
    """Get model performance data for a specific time period."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT model_id, product_category, timestamp, accuracy_score, 
                   mape_score, rmse_score, prediction_count
            FROM model_performance 
            WHERE model_id = ? AND timestamp BETWEEN ? AND ?
            ORDER BY timestamp
        """, (model_id, start_date, end_date))
        
        rows = cursor.fetchall()
        conn.close()
        
        performance_data = []
        for row in rows:
            # Handle datetime parsing
            timestamp = row['timestamp']
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp)
            
            performance_data.append({
                'model_id': row['model_id'],
                'product_category': row['product_category'],
                'timestamp': timestamp,
                'accuracy_score': row['accuracy_score'],
                'mape_score': row['mape_score'],
                'rmse_score': row['rmse_score'],
                'prediction_count': row['prediction_count']
            })
        
        return performance_data
    except sqlite3.Error as e:
        logging.error(f"Error retrieving model performance data: {e}")
        return []


def _generate_ab_test_recommendation(
    is_statistically_significant: bool,
    is_practically_significant: bool,
    winner: Optional[str]
) -> str:
    """Generate recommendation based on A/B test results."""
    if is_statistically_significant and is_practically_significant:
        return f"Deploy {winner} - statistically and practically significant improvement"
    elif is_statistically_significant and not is_practically_significant:
        return "Continue testing - statistically significant but improvement may not be practically meaningful"
    elif not is_statistically_significant and is_practically_significant:
        return "Extend test duration - practically significant difference but needs more data for statistical confidence"
    else:
        return "No significant difference detected - continue with current model or investigate other improvements"


def generate_validation_report(
    model_id: str,
    validation_results: List[ValidationResult],
    include_recommendations: bool = True,
    include_statistical_analysis: bool = True,
    benchmark_models: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Generate comprehensive validation report for a model.
    
    Provides detailed analysis including:
    - Statistical validation summaries
    - Trend analysis and forecasting
    - Comparative analysis with benchmark models
    - Risk assessment and confidence intervals
    - Actionable recommendations with priority levels
    """
    try:
        if not validation_results:
            return {
                "success": False,
                "message": "No validation results available for report generation"
            }
        
        # Basic summary statistics
        total_validations = len(validation_results)
        passed_validations = sum(1 for r in validation_results if r.passed)
        failed_validations = total_validations - passed_validations
        success_rate = passed_validations / total_validations if total_validations > 0 else 0
        
        # Extract metrics for analysis
        accuracy_scores = [r.accuracy_score for r in validation_results if r.accuracy_score is not None]
        baseline_accuracies = [r.baseline_accuracy for r in validation_results if r.baseline_accuracy is not None]
        improvements = [r.improvement_percentage for r in validation_results if r.improvement_percentage is not None]
        validation_dates = [r.validation_date for r in validation_results]
        
        # Advanced statistical analysis
        statistical_analysis = {}
        if include_statistical_analysis and len(accuracy_scores) >= 3:
            statistical_analysis = _perform_validation_statistical_analysis(
                accuracy_scores, baseline_accuracies, improvements
            )
        
        # Trend analysis with forecasting
        trend_analysis = _perform_trend_analysis(validation_results)
        
        # Risk assessment
        risk_assessment = _assess_validation_risk(validation_results, accuracy_scores, improvements)
        
        # Performance stability analysis
        stability_analysis = _analyze_performance_stability(validation_results)
        
        # Comparative analysis with benchmarks
        comparative_analysis = {}
        if benchmark_models:
            comparative_analysis = _perform_benchmark_comparison(
                model_id, validation_results, benchmark_models
            )
        
        # Generate prioritized recommendations
        recommendations = []
        if include_recommendations:
            recommendations = _generate_prioritized_recommendations(
                validation_results, statistical_analysis, trend_analysis, 
                risk_assessment, stability_analysis
            )
        
        # Validation quality assessment
        quality_assessment = _assess_validation_quality(validation_results)
        
        # Generate executive summary
        executive_summary = _generate_executive_summary(
            model_id, success_rate, trend_analysis, risk_assessment, recommendations
        )
        
        return {
            "success": True,
            "model_id": model_id,
            "report_generated_at": datetime.now().isoformat(),
            "report_version": "2.0",
            "executive_summary": executive_summary,
            "validation_period": {
                "earliest": min(validation_dates).isoformat(),
                "latest": max(validation_dates).isoformat(),
                "total_validations": total_validations,
                "duration_days": (max(validation_dates) - min(validation_dates)).days
            },
            "summary_statistics": {
                "passed_validations": passed_validations,
                "failed_validations": failed_validations,
                "success_rate": float(success_rate),
                "average_improvement": float(np.mean(improvements)) if improvements else 0,
                "consistency_score": float(1 - np.std(improvements) / np.mean(improvements)) if improvements and np.mean(improvements) > 0 else 0
            },
            "statistical_analysis": statistical_analysis,
            "trend_analysis": trend_analysis,
            "risk_assessment": risk_assessment,
            "stability_analysis": stability_analysis,
            "comparative_analysis": comparative_analysis,
            "quality_assessment": quality_assessment,
            "recommendations": recommendations,
            "detailed_results": [
                {
                    "validation_date": r.validation_date.isoformat(),
                    "dataset_id": r.validation_dataset_id,
                    "accuracy_score": r.accuracy_score,
                    "baseline_accuracy": r.baseline_accuracy,
                    "improvement_percentage": r.improvement_percentage,
                    "statistical_significance": r.statistical_significance,
                    "status": r.validation_status.value,
                    "passed": r.passed,
                    "notes": r.validation_notes
                }
                for r in sorted(validation_results, key=lambda x: x.validation_date, reverse=True)
            ],
            "metadata": {
                "report_type": "comprehensive_validation_report",
                "analysis_depth": "advanced" if include_statistical_analysis else "standard",
                "benchmark_comparison": len(benchmark_models) if benchmark_models else 0,
                "confidence_level": 0.95
            }
        }
        
    except Exception as e:
        logging.error(f"Error generating validation report: {e}")
        return {
            "success": False,
            "message": f"Error generating validation report: {str(e)}"
        }


def _perform_validation_statistical_analysis(
    accuracy_scores: List[float],
    baseline_accuracies: List[float],
    improvements: List[float]
) -> Dict[str, Any]:
    """Perform advanced statistical analysis on validation results."""
    try:
        # Descriptive statistics with confidence intervals
        def calculate_stats_with_ci(data, confidence=0.95):
            if not data:
                return {}
            
            data_array = np.array(data)
            n = len(data_array)
            mean = np.mean(data_array)
            std = np.std(data_array, ddof=1)
            se = std / np.sqrt(n)
            
            # Confidence interval
            from scipy.stats import t
            t_critical = t.ppf((1 + confidence) / 2, n - 1)
            ci_lower = mean - t_critical * se
            ci_upper = mean + t_critical * se
            
            return {
                "mean": float(mean),
                "std": float(std),
                "min": float(np.min(data_array)),
                "max": float(np.max(data_array)),
                "median": float(np.median(data_array)),
                "q25": float(np.percentile(data_array, 25)),
                "q75": float(np.percentile(data_array, 75)),
                "confidence_interval": [float(ci_lower), float(ci_upper)],
                "sample_size": n
            }
        
        accuracy_stats = calculate_stats_with_ci(accuracy_scores)
        baseline_stats = calculate_stats_with_ci(baseline_accuracies)
        improvement_stats = calculate_stats_with_ci(improvements)
        
        # Test for normality
        from scipy.stats import shapiro, jarque_bera
        
        normality_tests = {}
        for name, data in [("accuracy", accuracy_scores), ("improvements", improvements)]:
            if len(data) >= 3:
                shapiro_stat, shapiro_p = shapiro(data)
                normality_tests[name] = {
                    "shapiro_wilk": {
                        "statistic": float(shapiro_stat),
                        "p_value": float(shapiro_p),
                        "is_normal": shapiro_p > 0.05
                    }
                }
        
        # One-sample t-test against minimum improvement threshold
        improvement_test = {}
        if improvements:
            from scipy.stats import ttest_1samp
            t_stat, p_val = ttest_1samp(improvements, MIN_IMPROVEMENT_THRESHOLD)
            improvement_test = {
                "null_hypothesis": f"Mean improvement = {MIN_IMPROVEMENT_THRESHOLD}",
                "t_statistic": float(t_stat),
                "p_value": float(p_val),
                "significant": p_val < 0.05,
                "conclusion": "Significantly above threshold" if t_stat > 0 and p_val < 0.05 else "Not significantly above threshold"
            }
        
        return {
            "accuracy_statistics": accuracy_stats,
            "baseline_statistics": baseline_stats,
            "improvement_statistics": improvement_stats,
            "normality_tests": normality_tests,
            "threshold_test": improvement_test,
            "statistical_power": _calculate_validation_power(improvements) if improvements else {}
        }
        
    except Exception as e:
        return {"error": f"Statistical analysis failed: {e}"}


def _perform_trend_analysis(validation_results: List[ValidationResult]) -> Dict[str, Any]:
    """Perform comprehensive trend analysis with forecasting."""
    try:
        if len(validation_results) < 3:
            return {"error": "Insufficient data for trend analysis"}
        
        # Sort by date
        sorted_results = sorted(validation_results, key=lambda x: x.validation_date)
        
        # Extract time series data
        dates = [r.validation_date for r in sorted_results]
        accuracies = [r.accuracy_score for r in sorted_results if r.accuracy_score is not None]
        improvements = [r.improvement_percentage for r in sorted_results if r.improvement_percentage is not None]
        success_indicators = [1 if r.passed else 0 for r in sorted_results]
        
        # Convert dates to numeric for regression
        date_nums = [(d - dates[0]).days for d in dates]
        
        # Linear trend analysis
        trends = {}
        for name, values in [("accuracy", accuracies), ("improvement", improvements), ("success_rate", success_indicators)]:
            if len(values) >= 3:
                # Linear regression
                from scipy.stats import linregress
                slope, intercept, r_value, p_value, std_err = linregress(date_nums[:len(values)], values)
                
                trends[name] = {
                    "slope": float(slope),
                    "intercept": float(intercept),
                    "r_squared": float(r_value**2),
                    "p_value": float(p_value),
                    "significant_trend": p_value < 0.05,
                    "direction": "improving" if slope > 0 else "declining" if slope < 0 else "stable",
                    "strength": "strong" if abs(r_value) > 0.7 else "moderate" if abs(r_value) > 0.4 else "weak"
                }
        
        # Recent vs historical comparison
        recent_window = min(5, len(sorted_results) // 2)
        recent_results = sorted_results[-recent_window:]
        historical_results = sorted_results[:-recent_window] if len(sorted_results) > recent_window else []
        
        comparison = {}
        if historical_results:
            recent_success_rate = sum(1 for r in recent_results if r.passed) / len(recent_results)
            historical_success_rate = sum(1 for r in historical_results if r.passed) / len(historical_results)
            
            recent_improvements = [r.improvement_percentage for r in recent_results if r.improvement_percentage is not None]
            historical_improvements = [r.improvement_percentage for r in historical_results if r.improvement_percentage is not None]
            
            comparison = {
                "recent_success_rate": float(recent_success_rate),
                "historical_success_rate": float(historical_success_rate),
                "success_rate_change": float(recent_success_rate - historical_success_rate),
                "recent_avg_improvement": float(np.mean(recent_improvements)) if recent_improvements else 0,
                "historical_avg_improvement": float(np.mean(historical_improvements)) if historical_improvements else 0,
                "improvement_change": float(np.mean(recent_improvements) - np.mean(historical_improvements)) if recent_improvements and historical_improvements else 0
            }
        
        # Forecast next validation outcome
        forecast = {}
        if "accuracy" in trends and trends["accuracy"]["significant_trend"]:
            next_date_num = date_nums[-1] + 7  # Assume weekly validations
            predicted_accuracy = trends["accuracy"]["slope"] * next_date_num + trends["accuracy"]["intercept"]
            
            forecast = {
                "predicted_accuracy": float(predicted_accuracy),
                "confidence": "high" if trends["accuracy"]["r_squared"] > 0.7 else "medium" if trends["accuracy"]["r_squared"] > 0.4 else "low",
                "trend_based": True
            }
        
        return {
            "trends": trends,
            "recent_vs_historical": comparison,
            "forecast": forecast,
            "trend_summary": _summarize_trends(trends)
        }
        
    except Exception as e:
        return {"error": f"Trend analysis failed: {e}"}


def _assess_validation_risk(
    validation_results: List[ValidationResult],
    accuracy_scores: List[float],
    improvements: List[float]
) -> Dict[str, Any]:
    """Assess validation-related risks."""
    try:
        # Variability risk
        accuracy_cv = np.std(accuracy_scores) / np.mean(accuracy_scores) if accuracy_scores and np.mean(accuracy_scores) > 0 else 0
        improvement_cv = np.std(improvements) / abs(np.mean(improvements)) if improvements and np.mean(improvements) != 0 else 0
        
        # Failure pattern risk
        recent_failures = sum(1 for r in validation_results[-5:] if not r.passed)
        failure_risk = recent_failures / min(5, len(validation_results))
        
        # Threshold risk (how close to failing threshold)
        if improvements:
            avg_improvement = np.mean(improvements)
            threshold_margin = avg_improvement - MIN_IMPROVEMENT_THRESHOLD
            threshold_risk = max(0, 1 - (threshold_margin / MIN_IMPROVEMENT_THRESHOLD)) if MIN_IMPROVEMENT_THRESHOLD > 0 else 0
        else:
            threshold_risk = 1
        
        # Overall risk score
        risk_factors = [accuracy_cv, improvement_cv, failure_risk, threshold_risk]
        overall_risk = np.mean(risk_factors)
        
        risk_level = "high" if overall_risk > 0.7 else "medium" if overall_risk > 0.4 else "low"
        
        return {
            "overall_risk_score": float(overall_risk),
            "risk_level": risk_level,
            "risk_factors": {
                "accuracy_variability": {
                    "coefficient_of_variation": float(accuracy_cv),
                    "risk_level": "high" if accuracy_cv > 0.1 else "medium" if accuracy_cv > 0.05 else "low"
                },
                "improvement_variability": {
                    "coefficient_of_variation": float(improvement_cv),
                    "risk_level": "high" if improvement_cv > 0.5 else "medium" if improvement_cv > 0.2 else "low"
                },
                "recent_failure_rate": {
                    "rate": float(failure_risk),
                    "risk_level": "high" if failure_risk > 0.4 else "medium" if failure_risk > 0.2 else "low"
                },
                "threshold_proximity": {
                    "risk_score": float(threshold_risk),
                    "risk_level": "high" if threshold_risk > 0.7 else "medium" if threshold_risk > 0.4 else "low"
                }
            },
            "mitigation_recommendations": _generate_risk_mitigation_recommendations(risk_factors, risk_level)
        }
        
    except Exception as e:
        return {"error": f"Risk assessment failed: {e}"}


def _analyze_performance_stability(validation_results: List[ValidationResult]) -> Dict[str, Any]:
    """Analyze performance stability over time."""
    try:
        if len(validation_results) < 3:
            return {"error": "Insufficient data for stability analysis"}
        
        # Sort by date
        sorted_results = sorted(validation_results, key=lambda x: x.validation_date)
        
        # Calculate rolling statistics
        window_size = min(3, len(sorted_results))
        rolling_stats = []
        
        for i in range(window_size - 1, len(sorted_results)):
            window = sorted_results[i - window_size + 1:i + 1]
            window_accuracies = [r.accuracy_score for r in window if r.accuracy_score is not None]
            window_improvements = [r.improvement_percentage for r in window if r.improvement_percentage is not None]
            
            if window_accuracies:
                rolling_stats.append({
                    "date": window[-1].validation_date,
                    "mean_accuracy": np.mean(window_accuracies),
                    "std_accuracy": np.std(window_accuracies),
                    "mean_improvement": np.mean(window_improvements) if window_improvements else 0,
                    "success_rate": sum(1 for r in window if r.passed) / len(window)
                })
        
        # Stability metrics
        if len(rolling_stats) >= 2:
            accuracy_stds = [s["std_accuracy"] for s in rolling_stats]
            success_rates = [s["success_rate"] for s in rolling_stats]
            
            stability_score = 1 - (np.mean(accuracy_stds) / np.mean([s["mean_accuracy"] for s in rolling_stats]))
            consistency_score = 1 - np.std(success_rates)
            
            overall_stability = (stability_score + consistency_score) / 2
        else:
            overall_stability = 0.5  # Default moderate stability
        
        stability_level = "high" if overall_stability > 0.8 else "medium" if overall_stability > 0.6 else "low"
        
        return {
            "overall_stability_score": float(overall_stability),
            "stability_level": stability_level,
            "rolling_statistics": [
                {
                    "date": s["date"].isoformat(),
                    "mean_accuracy": float(s["mean_accuracy"]),
                    "std_accuracy": float(s["std_accuracy"]),
                    "mean_improvement": float(s["mean_improvement"]),
                    "success_rate": float(s["success_rate"])
                }
                for s in rolling_stats
            ],
            "stability_trends": {
                "accuracy_variance_trend": _calculate_trend([s["std_accuracy"] for s in rolling_stats]),
                "success_rate_trend": _calculate_trend([s["success_rate"] for s in rolling_stats])
            }
        }
        
    except Exception as e:
        return {"error": f"Stability analysis failed: {e}"}


def _perform_benchmark_comparison(
    model_id: str,
    validation_results: List[ValidationResult],
    benchmark_models: List[str]
) -> Dict[str, Any]:
    """Compare model performance against benchmarks."""
    try:
        # Get validation results for benchmark models
        benchmark_data = {}
        for benchmark_id in benchmark_models:
            benchmark_results = get_validation_results(benchmark_id)
            if benchmark_results:
                benchmark_data[benchmark_id] = {
                    "success_rate": sum(1 for r in benchmark_results if r.passed) / len(benchmark_results),
                    "avg_improvement": np.mean([r.improvement_percentage for r in benchmark_results if r.improvement_percentage is not None]),
                    "avg_accuracy": np.mean([r.accuracy_score for r in benchmark_results if r.accuracy_score is not None])
                }
        
        # Current model stats
        current_success_rate = sum(1 for r in validation_results if r.passed) / len(validation_results)
        current_avg_improvement = np.mean([r.improvement_percentage for r in validation_results if r.improvement_percentage is not None])
        current_avg_accuracy = np.mean([r.accuracy_score for r in validation_results if r.accuracy_score is not None])
        
        # Comparative analysis
        comparisons = {}
        for benchmark_id, benchmark_stats in benchmark_data.items():
            comparisons[benchmark_id] = {
                "success_rate_difference": float(current_success_rate - benchmark_stats["success_rate"]),
                "improvement_difference": float(current_avg_improvement - benchmark_stats["avg_improvement"]),
                "accuracy_difference": float(current_avg_accuracy - benchmark_stats["avg_accuracy"]),
                "overall_better": (
                    current_success_rate > benchmark_stats["success_rate"] and
                    current_avg_improvement > benchmark_stats["avg_improvement"] and
                    current_avg_accuracy > benchmark_stats["avg_accuracy"]
                )
            }
        
        # Ranking
        all_models = [(model_id, current_success_rate, current_avg_improvement, current_avg_accuracy)]
        for benchmark_id, stats in benchmark_data.items():
            all_models.append((benchmark_id, stats["success_rate"], stats["avg_improvement"], stats["avg_accuracy"]))
        
        # Sort by composite score
        ranked_models = sorted(all_models, key=lambda x: (x[1] + x[2] + x[3]) / 3, reverse=True)
        current_rank = next(i for i, (mid, _, _, _) in enumerate(ranked_models, 1) if mid == model_id)
        
        return {
            "benchmark_models": list(benchmark_models),
            "comparisons": comparisons,
            "ranking": {
                "current_rank": current_rank,
                "total_models": len(ranked_models),
                "percentile": float((len(ranked_models) - current_rank + 1) / len(ranked_models) * 100)
            },
            "performance_summary": {
                "better_than": sum(1 for comp in comparisons.values() if comp["overall_better"]),
                "worse_than": sum(1 for comp in comparisons.values() if not comp["overall_better"]),
                "competitive": current_rank <= len(ranked_models) // 2
            }
        }
        
    except Exception as e:
        return {"error": f"Benchmark comparison failed: {e}"}


def _generate_prioritized_recommendations(
    validation_results: List[ValidationResult],
    statistical_analysis: Dict[str, Any],
    trend_analysis: Dict[str, Any],
    risk_assessment: Dict[str, Any],
    stability_analysis: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Generate prioritized recommendations with action items."""
    recommendations = []
    
    # High priority recommendations
    success_rate = sum(1 for r in validation_results if r.passed) / len(validation_results)
    
    if success_rate < 0.5:
        recommendations.append({
            "priority": "high",
            "category": "performance",
            "title": "Critical Performance Issues",
            "description": f"Model validation success rate is {success_rate:.1%}, indicating fundamental performance problems",
            "action_items": [
                "Review model architecture and training methodology",
                "Investigate data quality and feature engineering",
                "Consider alternative algorithms or ensemble approaches",
                "Implement immediate performance monitoring"
            ],
            "timeline": "immediate"
        })
    
    if risk_assessment.get("risk_level") == "high":
        recommendations.append({
            "priority": "high",
            "category": "risk",
            "title": "High Risk Factors Detected",
            "description": "Multiple risk factors indicate potential deployment issues",
            "action_items": [
                "Implement enhanced monitoring and alerting",
                "Consider gradual rollout strategy",
                "Establish rollback procedures",
                "Increase validation frequency"
            ],
            "timeline": "immediate"
        })
    
    # Medium priority recommendations
    if trend_analysis.get("trends", {}).get("accuracy", {}).get("direction") == "declining":
        recommendations.append({
            "priority": "medium",
            "category": "trend",
            "title": "Declining Performance Trend",
            "description": "Model accuracy shows declining trend over time",
            "action_items": [
                "Investigate causes of performance degradation",
                "Review recent data changes or drift",
                "Consider model retraining or updates",
                "Implement trend monitoring"
            ],
            "timeline": "within_week"
        })
    
    if stability_analysis.get("stability_level") == "low":
        recommendations.append({
            "priority": "medium",
            "category": "stability",
            "title": "Performance Instability",
            "description": "Model shows inconsistent validation performance",
            "action_items": [
                "Analyze validation methodology consistency",
                "Review data preprocessing stability",
                "Consider ensemble methods for stability",
                "Implement performance variance monitoring"
            ],
            "timeline": "within_week"
        })
    
    # Low priority recommendations
    improvements = [r.improvement_percentage for r in validation_results if r.improvement_percentage is not None]
    if improvements and np.mean(improvements) < MIN_IMPROVEMENT_THRESHOLD * 2:
        recommendations.append({
            "priority": "low",
            "category": "optimization",
            "title": "Marginal Performance Improvements",
            "description": f"Average improvement ({np.mean(improvements):.1%}) is close to minimum threshold",
            "action_items": [
                "Explore advanced feature engineering",
                "Consider hyperparameter optimization",
                "Investigate ensemble methods",
                "Review business impact thresholds"
            ],
            "timeline": "within_month"
        })
    
    # Positive recommendations
    if success_rate >= 0.8 and risk_assessment.get("risk_level") == "low":
        recommendations.append({
            "priority": "low",
            "category": "deployment",
            "title": "Ready for Production Deployment",
            "description": "Model shows consistent validation success with low risk",
            "action_items": [
                "Proceed with production deployment",
                "Implement standard monitoring",
                "Document deployment procedures",
                "Plan regular validation schedule"
            ],
            "timeline": "ready"
        })
    
    return sorted(recommendations, key=lambda x: {"high": 3, "medium": 2, "low": 1}[x["priority"]], reverse=True)


def _generate_executive_summary(
    model_id: str,
    success_rate: float,
    trend_analysis: Dict[str, Any],
    risk_assessment: Dict[str, Any],
    recommendations: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Generate executive summary for validation report."""
    # Overall status
    if success_rate >= 0.8 and risk_assessment.get("risk_level") == "low":
        status = "excellent"
        status_description = "Model demonstrates excellent validation performance with low risk"
    elif success_rate >= 0.6 and risk_assessment.get("risk_level") in ["low", "medium"]:
        status = "good"
        status_description = "Model shows good validation performance with manageable risk"
    elif success_rate >= 0.4:
        status = "concerning"
        status_description = "Model validation performance requires attention and improvement"
    else:
        status = "critical"
        status_description = "Model validation performance is critically low and requires immediate action"
    
    # Key metrics
    key_metrics = {
        "validation_success_rate": f"{success_rate:.1%}",
        "risk_level": risk_assessment.get("risk_level", "unknown"),
        "trend_direction": trend_analysis.get("trends", {}).get("accuracy", {}).get("direction", "unknown"),
        "high_priority_actions": len([r for r in recommendations if r.get("priority") == "high"])
    }
    
    # Key insights
    insights = []
    if success_rate >= 0.8:
        insights.append("Strong validation performance indicates model reliability")
    if trend_analysis.get("trends", {}).get("accuracy", {}).get("direction") == "improving":
        insights.append("Performance trend is positive and improving over time")
    if risk_assessment.get("risk_level") == "low":
        insights.append("Low risk profile supports confident deployment")
    
    # Next steps
    high_priority_recs = [r for r in recommendations if r.get("priority") == "high"]
    next_steps = [rec["title"] for rec in high_priority_recs[:3]]  # Top 3 high priority items
    
    return {
        "overall_status": status,
        "status_description": status_description,
        "key_metrics": key_metrics,
        "key_insights": insights,
        "immediate_next_steps": next_steps,
        "deployment_recommendation": _get_deployment_recommendation(status, success_rate, risk_assessment)
    }


def _get_deployment_recommendation(status: str, success_rate: float, risk_assessment: Dict[str, Any]) -> str:
    """Get deployment recommendation based on validation analysis."""
    if status == "excellent":
        return "Recommend immediate production deployment"
    elif status == "good":
        return "Recommend deployment with standard monitoring"
    elif status == "concerning":
        return "Recommend additional validation before deployment"
    else:
        return "Do not recommend deployment - requires significant improvements"


# Helper functions for statistical calculations
def _calculate_validation_power(improvements: List[float]) -> Dict[str, Any]:
    """Calculate statistical power for validation tests."""
    try:
        if len(improvements) < 3:
            return {}
        
        from scipy.stats import ttest_1samp, norm
        
        # Observed effect size
        mean_improvement = np.mean(improvements)
        std_improvement = np.std(improvements, ddof=1)
        effect_size = (mean_improvement - MIN_IMPROVEMENT_THRESHOLD) / std_improvement if std_improvement > 0 else 0
        
        # Power calculation
        n = len(improvements)
        alpha = 0.05
        z_alpha = norm.ppf(1 - alpha/2)
        z_beta = norm.ppf(0.8)  # 80% power
        
        # Observed power
        ncp = effect_size * np.sqrt(n)
        power = 1 - norm.cdf(z_alpha - ncp) + norm.cdf(-z_alpha - ncp)
        
        return {
            "observed_power": float(power),
            "effect_size": float(effect_size),
            "sample_size": n,
            "adequate_power": power >= 0.8,
            "recommended_sample_size": int(((z_alpha + z_beta) / effect_size)**2) if effect_size > 0 else None
        }
        
    except Exception:
        return {}


def _calculate_trend(values: List[float]) -> str:
    """Calculate trend direction from a series of values."""
    if len(values) < 2:
        return "insufficient_data"
    
    from scipy.stats import linregress
    x = list(range(len(values)))
    slope, _, r_value, p_value, _ = linregress(x, values)
    
    if p_value < 0.05:
        if slope > 0:
            return "improving"
        else:
            return "declining"
    else:
        return "stable"


def _summarize_trends(trends: Dict[str, Any]) -> str:
    """Summarize trend analysis results."""
    trend_directions = [trend.get("direction", "unknown") for trend in trends.values()]
    
    if all(d == "improving" for d in trend_directions):
        return "All metrics show improving trends"
    elif all(d == "declining" for d in trend_directions):
        return "All metrics show declining trends"
    elif "declining" in trend_directions:
        return "Mixed trends with some declining metrics"
    elif "improving" in trend_directions:
        return "Mixed trends with some improving metrics"
    else:
        return "Stable trends across metrics"


def _generate_risk_mitigation_recommendations(risk_factors: List[float], risk_level: str) -> List[str]:
    """Generate risk mitigation recommendations."""
    recommendations = []
    
    if risk_level == "high":
        recommendations.extend([
            "Implement enhanced monitoring and alerting systems",
            "Consider gradual rollout with canary deployment",
            "Establish clear rollback procedures and criteria",
            "Increase validation frequency and rigor"
        ])
    elif risk_level == "medium":
        recommendations.extend([
            "Implement standard monitoring procedures",
            "Consider A/B testing for deployment validation",
            "Establish performance baselines and thresholds"
        ])
    
    return recommendations


def _assess_validation_quality(validation_results: List[ValidationResult]) -> Dict[str, Any]:
    """Assess the quality of validation methodology and data."""
    try:
        # Data completeness
        total_validations = len(validation_results)
        complete_validations = sum(
            1 for r in validation_results 
            if r.accuracy_score is not None and r.baseline_accuracy is not None
        )
        completeness_score = complete_validations / total_validations if total_validations > 0 else 0
        
        # Temporal distribution
        dates = [r.validation_date for r in validation_results]
        if len(dates) > 1:
            date_spans = [(dates[i] - dates[i-1]).days for i in range(1, len(dates))]
            avg_interval = np.mean(date_spans)
            interval_consistency = 1 - (np.std(date_spans) / avg_interval) if avg_interval > 0 else 0
        else:
            interval_consistency = 1
        
        # Statistical significance coverage
        sig_tests = sum(1 for r in validation_results if r.statistical_significance is not None)
        sig_coverage = sig_tests / total_validations if total_validations > 0 else 0
        
        # Overall quality score
        quality_score = (completeness_score + interval_consistency + sig_coverage) / 3
        quality_level = "high" if quality_score > 0.8 else "medium" if quality_score > 0.6 else "low"
        
        return {
            "overall_quality_score": float(quality_score),
            "quality_level": quality_level,
            "components": {
                "data_completeness": float(completeness_score),
                "temporal_consistency": float(interval_consistency),
                "statistical_coverage": float(sig_coverage)
            },
            "recommendations": _get_quality_recommendations(quality_level, completeness_score, interval_consistency, sig_coverage)
        }
        
    except Exception as e:
        return {"error": f"Quality assessment failed: {e}"}


def _get_quality_recommendations(
    quality_level: str, 
    completeness: float, 
    consistency: float, 
    coverage: float
) -> List[str]:
    """Get recommendations for improving validation quality."""
    recommendations = []
    
    if completeness < 0.8:
        recommendations.append("Improve data collection to ensure complete validation metrics")
    
    if consistency < 0.7:
        recommendations.append("Establish regular validation schedule for consistent temporal coverage")
    
    if coverage < 0.8:
        recommendations.append("Include statistical significance testing in all validations")
    
    if quality_level == "low":
        recommendations.append("Comprehensive review of validation methodology needed")
    
    return recommendations


# --- MCP Server Setup ---
server = Server("model-validation-mcp-server")

@server.list_tools()
async def handle_list_tools() -> List[mcp_types.Tool]:
    """List available model validation tools."""
    return [
        mcp_types.Tool(
            name="create_holdout_dataset",
            description="Create a holdout dataset for model validation testing",
            inputSchema={
                "type": "object",
                "properties": {
                    "product_categories": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of product categories to include"
                    },
                    "date_range_start": {
                        "type": "string",
                        "description": "Start date for data range (ISO format)"
                    },
                    "date_range_end": {
                        "type": "string",
                        "description": "End date for data range (ISO format)"
                    },
                    "sample_size": {
                        "type": "integer",
                        "description": "Optional sample size (defaults to 20% of available data)"
                    }
                },
                "required": ["product_categories", "date_range_start", "date_range_end"]
            }
        ),
        mcp_types.Tool(
            name="validate_model_performance",
            description="Validate model performance against holdout dataset and baseline",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": {
                        "type": "string",
                        "description": "Model identifier to validate"
                    },
                    "holdout_dataset_id": {
                        "type": "string",
                        "description": "Holdout dataset identifier for testing"
                    },
                    "test_accuracy": {
                        "type": "number",
                        "description": "Model accuracy score on test data"
                    },
                    "test_mape": {
                        "type": "number",
                        "description": "Optional MAPE score on test data"
                    },
                    "test_rmse": {
                        "type": "number",
                        "description": "Optional RMSE score on test data"
                    }
                },
                "required": ["model_id", "holdout_dataset_id", "test_accuracy"]
            }
        ),
        mcp_types.Tool(
            name="compare_models_statistically",
            description="Perform statistical significance testing between two models",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_a_results": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Performance results for model A"
                    },
                    "model_b_results": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Performance results for model B"
                    },
                    "alpha": {
                        "type": "number",
                        "description": "Significance level (default 0.05)"
                    }
                },
                "required": ["model_a_results", "model_b_results"]
            }
        ),
        mcp_types.Tool(
            name="get_validation_results",
            description="Retrieve validation results for a model",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": {
                        "type": "string",
                        "description": "Model identifier"
                    },
                    "days_back": {
                        "type": "integer",
                        "description": "Number of days back to retrieve results (default 30)"
                    }
                },
                "required": ["model_id"]
            }
        ),
        mcp_types.Tool(
            name="get_holdout_dataset_info",
            description="Get information about a holdout dataset",
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset_id": {
                        "type": "string",
                        "description": "Holdout dataset identifier"
                    }
                },
                "required": ["dataset_id"]
            }
        ),
        mcp_types.Tool(
            name="perform_ab_testing",
            description="Perform A/B testing analysis between two models",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_a_id": {
                        "type": "string",
                        "description": "First model identifier for comparison"
                    },
                    "model_b_id": {
                        "type": "string",
                        "description": "Second model identifier for comparison"
                    },
                    "test_duration_days": {
                        "type": "integer",
                        "description": "Duration of test period in days (default 7)"
                    },
                    "confidence_level": {
                        "type": "number",
                        "description": "Statistical confidence level (default 0.95)"
                    }
                },
                "required": ["model_a_id", "model_b_id"]
            }
        ),
        mcp_types.Tool(
            name="generate_validation_report",
            description="Generate comprehensive validation report for a model",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": {
                        "type": "string",
                        "description": "Model identifier"
                    },
                    "days_back": {
                        "type": "integer",
                        "description": "Number of days back to include in report (default 30)"
                    },
                    "include_recommendations": {
                        "type": "boolean",
                        "description": "Whether to include recommendations (default true)"
                    }
                },
                "required": ["model_id"]
            }
        ),
        mcp_types.Tool(
            name="calculate_business_impact",
            description="Calculate comprehensive business impact of model improvements",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": {
                        "type": "string",
                        "description": "Model identifier"
                    },
                    "deployment_id": {
                        "type": "string",
                        "description": "Deployment identifier for tracking"
                    },
                    "baseline_accuracy": {
                        "type": "number",
                        "description": "Baseline model accuracy"
                    },
                    "improved_accuracy": {
                        "type": "number",
                        "description": "Improved model accuracy"
                    },
                    "product_categories": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of affected product categories"
                    },
                    "time_period_days": {
                        "type": "integer",
                        "description": "Time period for impact calculation in days (default 30)"
                    }
                },
                "required": ["model_id", "deployment_id", "baseline_accuracy", "improved_accuracy", "product_categories"]
            }
        ),
        mcp_types.Tool(
            name="create_approval_workflow",
            description="Create automated approval workflow based on validation results and business impact",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": {
                        "type": "string",
                        "description": "Model identifier"
                    },
                    "validation_result": {
                        "type": "object",
                        "description": "Validation result object with accuracy and statistical metrics",
                        "properties": {
                            "accuracy_score": {"type": "number"},
                            "baseline_accuracy": {"type": "number"},
                            "improvement_percentage": {"type": "number"},
                            "statistical_significance": {"type": "number"},
                            "validation_status": {"type": "string"}
                        },
                        "required": ["accuracy_score", "improvement_percentage"]
                    },
                    "business_impact": {
                        "type": "object",
                        "description": "Business impact calculation result"
                    },
                    "approval_thresholds": {
                        "type": "object",
                        "description": "Optional custom approval thresholds",
                        "properties": {
                            "min_accuracy_improvement": {"type": "number"},
                            "min_statistical_significance": {"type": "number"},
                            "min_business_impact": {"type": "number"},
                            "min_roi_percentage": {"type": "number"},
                            "max_risk_level": {"type": "string"}
                        }
                    }
                },
                "required": ["model_id", "validation_result", "business_impact"]
            }
        ),
        mcp_types.Tool(
            name="track_validation_audit_trail",
            description="Track validation and approval audit trail for compliance",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": {
                        "type": "string",
                        "description": "Model identifier"
                    },
                    "validation_id": {
                        "type": "integer",
                        "description": "Validation record identifier"
                    },
                    "action": {
                        "type": "string",
                        "description": "Action being tracked (e.g., 'validation_completed', 'approval_granted', 'deployment_authorized')"
                    },
                    "user_id": {
                        "type": "string",
                        "description": "Optional user identifier for manual actions"
                    },
                    "notes": {
                        "type": "string",
                        "description": "Optional notes about the action"
                    },
                    "metadata": {
                        "type": "object",
                        "description": "Optional additional metadata"
                    }
                },
                "required": ["model_id", "validation_id", "action"]
            }
        ),
        mcp_types.Tool(
            name="get_validation_audit_trail",
            description="Retrieve validation audit trail records",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": {
                        "type": "string",
                        "description": "Optional model identifier filter"
                    },
                    "validation_id": {
                        "type": "integer",
                        "description": "Optional validation record identifier filter"
                    },
                    "days_back": {
                        "type": "integer",
                        "description": "Number of days back to retrieve records (default 30)"
                    },
                    "action_filter": {
                        "type": "string",
                        "description": "Optional action type filter"
                    }
                }
            }
        )
    ]


@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> List[mcp_types.TextContent]:
    """Handle tool calls for model validation operations."""
    
    if name == "create_holdout_dataset":
        product_categories = arguments.get("product_categories", [])
        date_range_start = datetime.fromisoformat(arguments["date_range_start"])
        date_range_end = datetime.fromisoformat(arguments["date_range_end"])
        sample_size = arguments.get("sample_size")
        
        result = create_holdout_dataset(
            product_categories, date_range_start, date_range_end, sample_size
        )
        
        return [mcp_types.TextContent(
            type="text",
            text=json.dumps(result, indent=2)
        )]
    
    elif name == "validate_model_performance":
        model_id = arguments["model_id"]
        holdout_dataset_id = arguments["holdout_dataset_id"]
        test_accuracy = arguments["test_accuracy"]
        test_mape = arguments.get("test_mape")
        test_rmse = arguments.get("test_rmse")
        
        result = validate_model_performance(
            model_id, holdout_dataset_id, test_accuracy, test_mape, test_rmse
        )
        
        return [mcp_types.TextContent(
            type="text",
            text=json.dumps(result, indent=2)
        )]
    
    elif name == "compare_models_statistically":
        model_a_results = arguments["model_a_results"]
        model_b_results = arguments["model_b_results"]
        alpha = arguments.get("alpha", STATISTICAL_SIGNIFICANCE_THRESHOLD)
        
        result = compare_models_statistical_significance(
            model_a_results, model_b_results, alpha
        )
        
        return [mcp_types.TextContent(
            type="text",
            text=json.dumps(result, indent=2)
        )]
    
    elif name == "get_validation_results":
        model_id = arguments["model_id"]
        days_back = arguments.get("days_back", VALIDATION_WINDOW_DAYS)
        
        results = get_validation_results(model_id, days_back)
        
        # Convert to serializable format
        results_data = []
        for result in results:
            results_data.append({
                "model_id": result.model_id,
                "validation_dataset_id": result.validation_dataset_id,
                "validation_date": result.validation_date.isoformat(),
                "accuracy_score": result.accuracy_score,
                "baseline_accuracy": result.baseline_accuracy,
                "improvement_percentage": result.improvement_percentage,
                "statistical_significance": result.statistical_significance,
                "validation_status": result.validation_status.value,
                "validation_notes": result.validation_notes,
                "passed": result.passed
            })
        
        return [mcp_types.TextContent(
            type="text",
            text=json.dumps({
                "success": True,
                "model_id": model_id,
                "validation_results": results_data,
                "total_results": len(results_data)
            }, indent=2)
        )]
    
    elif name == "get_holdout_dataset_info":
        dataset_id = arguments["dataset_id"]
        
        dataset = get_holdout_dataset(dataset_id)
        
        if dataset:
            dataset_info = {
                "success": True,
                "dataset_id": dataset.dataset_id,
                "product_categories": dataset.product_categories,
                "date_range_start": dataset.date_range_start.isoformat() if dataset.date_range_start else None,
                "date_range_end": dataset.date_range_end.isoformat() if dataset.date_range_end else None,
                "sample_size": dataset.sample_size,
                "dataset_path": dataset.dataset_path,
                "created_at": dataset.created_at.isoformat()
            }
        else:
            dataset_info = {
                "success": False,
                "message": f"Holdout dataset {dataset_id} not found"
            }
        
        return [mcp_types.TextContent(
            type="text",
            text=json.dumps(dataset_info, indent=2)
        )]
    
    elif name == "perform_ab_testing":
        model_a_id = arguments["model_a_id"]
        model_b_id = arguments["model_b_id"]
        test_duration_days = arguments.get("test_duration_days", 7)
        confidence_level = arguments.get("confidence_level", 0.95)
        
        result = perform_ab_testing_analysis(
            model_a_id, model_b_id, test_duration_days, confidence_level
        )
        
        return [mcp_types.TextContent(
            type="text",
            text=json.dumps(result, indent=2)
        )]
    
    elif name == "generate_validation_report":
        model_id = arguments["model_id"]
        days_back = arguments.get("days_back", VALIDATION_WINDOW_DAYS)
        include_recommendations = arguments.get("include_recommendations", True)
        
        # Get validation results
        validation_results = get_validation_results(model_id, days_back)
        
        # Generate report
        report = generate_validation_report(model_id, validation_results, include_recommendations)
        
        return [mcp_types.TextContent(
            type="text",
            text=json.dumps(report, indent=2)
        )]
    
    elif name == "calculate_business_impact":
        model_id = arguments["model_id"]
        deployment_id = arguments["deployment_id"]
        baseline_accuracy = arguments["baseline_accuracy"]
        improved_accuracy = arguments["improved_accuracy"]
        product_categories = arguments["product_categories"]
        time_period_days = arguments.get("time_period_days", 30)
        
        result = calculate_business_impact(
            model_id, deployment_id, baseline_accuracy, improved_accuracy,
            product_categories, time_period_days
        )
        
        return [mcp_types.TextContent(
            type="text",
            text=json.dumps(result, indent=2)
        )]
    
    elif name == "create_approval_workflow":
        model_id = arguments["model_id"]
        validation_result_data = arguments["validation_result"]
        business_impact = arguments["business_impact"]
        approval_thresholds = arguments.get("approval_thresholds")
        
        # Convert validation result data to ValidationResult object
        validation_result = ValidationResult(
            model_id=model_id,
            validation_dataset_id="temp_dataset",
            validation_date=datetime.now(),
            accuracy_score=validation_result_data.get("accuracy_score"),
            baseline_accuracy=validation_result_data.get("baseline_accuracy"),
            improvement_percentage=validation_result_data.get("improvement_percentage"),
            statistical_significance=validation_result_data.get("statistical_significance"),
            validation_status=ValidationStatus(validation_result_data.get("validation_status", "PENDING"))
        )
        
        result = create_approval_workflow(
            model_id, validation_result, business_impact, approval_thresholds
        )
        
        return [mcp_types.TextContent(
            type="text",
            text=json.dumps(result, indent=2)
        )]
    
    elif name == "track_validation_audit_trail":
        model_id = arguments["model_id"]
        validation_id = arguments["validation_id"]
        action = arguments["action"]
        user_id = arguments.get("user_id")
        notes = arguments.get("notes")
        metadata = arguments.get("metadata")
        
        result = track_validation_audit_trail(
            model_id, validation_id, action, user_id, notes, metadata
        )
        
        return [mcp_types.TextContent(
            type="text",
            text=json.dumps(result, indent=2)
        )]
    
    elif name == "get_validation_audit_trail":
        model_id = arguments.get("model_id")
        validation_id = arguments.get("validation_id")
        days_back = arguments.get("days_back", 30)
        action_filter = arguments.get("action_filter")
        
        result = get_validation_audit_trail(
            model_id, validation_id, days_back, action_filter
        )
        
        return [mcp_types.TextContent(
            type="text",
            text=json.dumps(result, indent=2)
        )]
    
    else:
        return [mcp_types.TextContent(
            type="text",
            text=f"Unknown tool: {name}"
        )]


def _calculate_trend(values: List[float]) -> str:
    """Calculate trend direction from a list of values."""
    if len(values) < 2:
        return "insufficient_data"
    
    # Simple linear trend calculation
    n = len(values)
    x = list(range(n))
    
    # Calculate slope using least squares
    x_mean = sum(x) / n
    y_mean = sum(values) / n
    
    numerator = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(n))
    denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
    
    if denominator == 0:
        return "stable"
    
    slope = numerator / denominator
    
    if slope > 0.01:
        return "improving"
    elif slope < -0.01:
        return "declining"
    else:
        return "stable"


def _summarize_trends(trends: Dict[str, Any]) -> str:
    """Summarize trend analysis results."""
    trend_descriptions = []
    
    for metric, trend_data in trends.items():
        if isinstance(trend_data, dict) and "direction" in trend_data:
            direction = trend_data["direction"]
            strength = trend_data.get("strength", "unknown")
            trend_descriptions.append(f"{metric}: {direction} ({strength})")
    
    if not trend_descriptions:
        return "No significant trends detected"
    
    return "; ".join(trend_descriptions)


def _calculate_validation_power(improvements: List[float]) -> Dict[str, Any]:
    """Calculate statistical power for validation tests."""
    try:
        if len(improvements) < 3:
            return {"error": "Insufficient data for power analysis"}
        
        import numpy as np
        from scipy import stats
        
        # Calculate effect size (Cohen's d)
        mean_improvement = np.mean(improvements)
        std_improvement = np.std(improvements, ddof=1)
        
        if std_improvement == 0:
            return {"error": "No variance in improvements"}
        
        effect_size = mean_improvement / std_improvement
        
        # Calculate observed power for one-sample t-test
        n = len(improvements)
        alpha = 0.05
        
        # Non-centrality parameter
        ncp = effect_size * np.sqrt(n)
        
        # Critical t-value
        t_critical = stats.t.ppf(1 - alpha/2, n - 1)
        
        # Power calculation
        power = 1 - stats.t.cdf(t_critical, n - 1, ncp) + stats.t.cdf(-t_critical, n - 1, ncp)
        
        return {
            "observed_power": float(power),
            "effect_size": float(effect_size),
            "sample_size": n,
            "adequate_power": power >= 0.8
        }
        
    except Exception as e:
        return {"error": f"Power analysis failed: {e}"}


def _generate_executive_summary(
    model_id: str,
    success_rate: float,
    trend_analysis: Dict[str, Any],
    risk_assessment: Dict[str, Any],
    recommendations: List[Dict[str, Any]]
) -> str:
    """Generate executive summary for validation report."""
    summary_parts = []
    
    # Performance summary
    if success_rate >= 0.8:
        performance_desc = "excellent"
    elif success_rate >= 0.6:
        performance_desc = "good"
    elif success_rate >= 0.4:
        performance_desc = "moderate"
    else:
        performance_desc = "poor"
    
    summary_parts.append(f"Model {model_id} shows {performance_desc} validation performance with {success_rate:.1%} success rate.")
    
    # Trend summary
    trend_summary = trend_analysis.get("trend_summary", "No clear trends identified")
    summary_parts.append(f"Performance trends: {trend_summary}.")
    
    # Risk summary
    risk_level = risk_assessment.get("risk_level", "unknown")
    summary_parts.append(f"Risk assessment: {risk_level} risk level.")
    
    # Recommendation summary
    high_priority_recs = [r for r in recommendations if r.get("priority") == "high"]
    if high_priority_recs:
        summary_parts.append(f"{len(high_priority_recs)} high-priority recommendations require immediate attention.")
    
    return " ".join(summary_parts)


def _assess_validation_quality(validation_results: List[ValidationResult]) -> Dict[str, Any]:
    """Assess the quality of validation data and processes."""
    try:
        if not validation_results:
            return {"error": "No validation results to assess"}
        
        # Data completeness
        complete_results = sum(1 for r in validation_results if r.accuracy_score is not None and r.baseline_accuracy is not None)
        completeness_score = complete_results / len(validation_results)
        
        # Temporal coverage
        dates = [r.validation_date for r in validation_results]
        date_span = (max(dates) - min(dates)).days if len(dates) > 1 else 0
        temporal_score = min(date_span / 30, 1.0)  # Normalize to 30 days
        
        # Statistical rigor
        statistical_results = sum(1 for r in validation_results if r.statistical_significance is not None)
        statistical_score = statistical_results / len(validation_results)
        
        # Overall quality score
        quality_score = (completeness_score + temporal_score + statistical_score) / 3
        
        quality_level = "high" if quality_score > 0.8 else "medium" if quality_score > 0.6 else "low"
        
        return {
            "overall_quality_score": float(quality_score),
            "quality_level": quality_level,
            "completeness_score": float(completeness_score),
            "temporal_coverage_score": float(temporal_score),
            "statistical_rigor_score": float(statistical_score),
            "total_validations": len(validation_results),
            "date_span_days": date_span
        }
        
    except Exception as e:
        return {"error": f"Quality assessment failed: {e}"}


async def main():
    """Main server entry point."""
    # Use stdin/stdout for communication
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="model-validation-mcp-server",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


# --- MCP Server Setup ---
server = Server("model-validation-mcp-server")

@server.list_tools()
async def handle_list_tools() -> List[mcp_types.Tool]:
    """List available model validation tools."""
    return [
        mcp_types.Tool(
            name="create_holdout_dataset",
            description="Create a holdout dataset for model validation testing",
            inputSchema={
                "type": "object",
                "properties": {
                    "product_categories": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of product categories to include"
                    },
                    "date_range_start": {
                        "type": "string",
                        "description": "Start date for data range (ISO format)"
                    },
                    "date_range_end": {
                        "type": "string",
                        "description": "End date for data range (ISO format)"
                    },
                    "sample_size": {
                        "type": "integer",
                        "description": "Optional sample size (auto-calculated if not provided)"
                    }
                },
                "required": ["product_categories", "date_range_start", "date_range_end"]
            }
        ),
        mcp_types.Tool(
            name="validate_model_performance",
            description="Validate model performance against holdout dataset with statistical significance testing",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": {
                        "type": "string",
                        "description": "Model identifier to validate"
                    },
                    "holdout_dataset_id": {
                        "type": "string",
                        "description": "Holdout dataset identifier for validation"
                    },
                    "test_accuracy": {
                        "type": "number",
                        "description": "Test accuracy score (0.0 to 1.0)"
                    },
                    "test_mape": {
                        "type": "number",
                        "description": "Optional MAPE score"
                    },
                    "test_rmse": {
                        "type": "number",
                        "description": "Optional RMSE score"
                    }
                },
                "required": ["model_id", "holdout_dataset_id", "test_accuracy"]
            }
        ),
        mcp_types.Tool(
            name="compare_models_statistical_significance",
            description="Perform comprehensive statistical significance testing between two models",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_a_results": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Performance results for model A"
                    },
                    "model_b_results": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Performance results for model B"
                    },
                    "alpha": {
                        "type": "number",
                        "description": "Significance level (default 0.05)",
                        "default": 0.05
                    }
                },
                "required": ["model_a_results", "model_b_results"]
            }
        ),
        mcp_types.Tool(
            name="perform_ab_testing_analysis",
            description="Perform comprehensive A/B testing analysis between two models with advanced statistical methods",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_a_id": {
                        "type": "string",
                        "description": "Model A identifier"
                    },
                    "model_b_id": {
                        "type": "string",
                        "description": "Model B identifier"
                    },
                    "test_duration_days": {
                        "type": "integer",
                        "description": "Test duration in days (default 7)",
                        "default": 7
                    },
                    "confidence_level": {
                        "type": "number",
                        "description": "Confidence level (default 0.95)",
                        "default": 0.95
                    },
                    "minimum_sample_size": {
                        "type": "integer",
                        "description": "Minimum sample size per model (default 10)",
                        "default": 10
                    },
                    "stratify_by_category": {
                        "type": "boolean",
                        "description": "Whether to perform stratified analysis by product category",
                        "default": true
                    }
                },
                "required": ["model_a_id", "model_b_id"]
            }
        ),
        mcp_types.Tool(
            name="generate_validation_report",
            description="Generate comprehensive validation report with statistical analysis, trends, and recommendations",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": {
                        "type": "string",
                        "description": "Model identifier for report generation"
                    },
                    "include_recommendations": {
                        "type": "boolean",
                        "description": "Include actionable recommendations",
                        "default": true
                    },
                    "include_statistical_analysis": {
                        "type": "boolean",
                        "description": "Include advanced statistical analysis",
                        "default": true
                    },
                    "benchmark_models": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of benchmark model IDs for comparison"
                    }
                },
                "required": ["model_id"]
            }
        ),
        mcp_types.Tool(
            name="get_validation_results",
            description="Retrieve validation results for a model within specified time window",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": {
                        "type": "string",
                        "description": "Model identifier"
                    },
                    "days_back": {
                        "type": "integer",
                        "description": "Number of days to look back (default 30)",
                        "default": 30
                    }
                },
                "required": ["model_id"]
            }
        ),
        mcp_types.Tool(
            name="get_baseline_model_accuracy",
            description="Get baseline model accuracy for comparison purposes",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": {
                        "type": "string",
                        "description": "Model identifier"
                    },
                    "product_categories": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Product categories for baseline calculation"
                    }
                },
                "required": ["model_id", "product_categories"]
            }
        )
    ]


@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> List[mcp_types.TextContent]:
    """Handle tool calls for model validation operations."""
    try:
        if name == "create_holdout_dataset":
            categories = arguments["product_categories"]
            start_date = datetime.fromisoformat(arguments["date_range_start"])
            end_date = datetime.fromisoformat(arguments["date_range_end"])
            sample_size = arguments.get("sample_size")
            
            result = create_holdout_dataset(categories, start_date, end_date, sample_size)
            return [mcp_types.TextContent(type="text", text=json.dumps(result, indent=2))]
        
        elif name == "validate_model_performance":
            model_id = arguments["model_id"]
            dataset_id = arguments["holdout_dataset_id"]
            test_accuracy = arguments["test_accuracy"]
            test_mape = arguments.get("test_mape")
            test_rmse = arguments.get("test_rmse")
            
            result = validate_model_performance(model_id, dataset_id, test_accuracy, test_mape, test_rmse)
            return [mcp_types.TextContent(type="text", text=json.dumps(result, indent=2))]
        
        elif name == "compare_models_statistical_significance":
            model_a_results = arguments["model_a_results"]
            model_b_results = arguments["model_b_results"]
            alpha = arguments.get("alpha", 0.05)
            
            result = compare_models_statistical_significance(model_a_results, model_b_results, alpha)
            return [mcp_types.TextContent(type="text", text=json.dumps(result, indent=2))]
        
        elif name == "perform_ab_testing_analysis":
            model_a_id = arguments["model_a_id"]
            model_b_id = arguments["model_b_id"]
            test_duration_days = arguments.get("test_duration_days", 7)
            confidence_level = arguments.get("confidence_level", 0.95)
            minimum_sample_size = arguments.get("minimum_sample_size", 10)
            stratify_by_category = arguments.get("stratify_by_category", True)
            
            result = perform_ab_testing_analysis(
                model_a_id, model_b_id, test_duration_days, confidence_level,
                minimum_sample_size, stratify_by_category
            )
            return [mcp_types.TextContent(type="text", text=json.dumps(result, indent=2))]
        
        elif name == "generate_validation_report":
            model_id = arguments["model_id"]
            include_recommendations = arguments.get("include_recommendations", True)
            include_statistical_analysis = arguments.get("include_statistical_analysis", True)
            benchmark_models = arguments.get("benchmark_models")
            
            # Get validation results for the model
            validation_results = get_validation_results(model_id)
            
            result = generate_validation_report(
                model_id, validation_results, include_recommendations,
                include_statistical_analysis, benchmark_models
            )
            return [mcp_types.TextContent(type="text", text=json.dumps(result, indent=2))]
        
        elif name == "get_validation_results":
            model_id = arguments["model_id"]
            days_back = arguments.get("days_back", 30)
            
            results = get_validation_results(model_id, days_back)
            
            # Convert ValidationResult objects to dictionaries
            results_dict = [
                {
                    "model_id": r.model_id,
                    "validation_dataset_id": r.validation_dataset_id,
                    "validation_date": r.validation_date.isoformat(),
                    "accuracy_score": r.accuracy_score,
                    "baseline_accuracy": r.baseline_accuracy,
                    "improvement_percentage": r.improvement_percentage,
                    "statistical_significance": r.statistical_significance,
                    "validation_status": r.validation_status.value,
                    "passed": r.passed,
                    "validation_notes": r.validation_notes
                }
                for r in results
            ]
            
            return [mcp_types.TextContent(type="text", text=json.dumps({
                "success": True,
                "results": results_dict,
                "count": len(results_dict)
            }, indent=2))]
        
        elif name == "get_baseline_model_accuracy":
            model_id = arguments["model_id"]
            product_categories = arguments["product_categories"]
            
            baseline = get_baseline_model_accuracy(model_id, product_categories)
            
            return [mcp_types.TextContent(type="text", text=json.dumps({
                "success": True,
                "baseline_accuracy": baseline,
                "model_id": model_id,
                "categories": product_categories
            }, indent=2))]
        
        else:
            return [mcp_types.TextContent(
                type="text", 
                text=json.dumps({"error": f"Unknown tool: {name}"})
            )]
    
    except Exception as e:
        logging.error(f"Error handling tool call {name}: {e}")
        return [mcp_types.TextContent(
            type="text", 
            text=json.dumps({"error": f"Tool execution failed: {str(e)}"})
        )]


async def main():
    """Main entry point for the MCP server."""
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="model-validation-mcp-server",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


if __name__ == "__main__":
    asyncio.run(main())