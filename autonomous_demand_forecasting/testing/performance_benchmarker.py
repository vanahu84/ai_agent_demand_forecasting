"""
Performance Benchmarking and Validation Tools

This module provides comprehensive performance benchmarking capabilities
for validating model accuracy and system performance in the autonomous
demand forecasting system.
"""

import time
import asyncio
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil
import tracemalloc

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Performance metrics for benchmarking"""
    accuracy_score: float
    mape_score: float
    rmse_score: float
    mae_score: float
    execution_time: float
    memory_usage: float
    cpu_usage: float
    throughput: float
    latency_p50: float
    latency_p95: float
    latency_p99: float

@dataclass
class BenchmarkConfig:
    """Configuration for performance benchmarking"""
    test_duration_seconds: int = 300
    concurrent_requests: int = 10
    warmup_requests: int = 50
    memory_profiling: bool = True
    cpu_profiling: bool = True
    accuracy_threshold: float = 0.85
    latency_threshold_ms: float = 1000
    throughput_threshold: float = 100  # requests per second

@dataclass
class AccuracyBenchmark:
    """Accuracy benchmark results"""
    model_name: str
    dataset_name: str
    accuracy_metrics: PerformanceMetrics
    prediction_errors: List[float]
    feature_importance: Dict[str, float]
    confusion_matrix: Optional[np.ndarray] = None

@dataclass
class SystemBenchmark:
    """System performance benchmark results"""
    component_name: str
    benchmark_config: BenchmarkConfig
    performance_metrics: PerformanceMetrics
    resource_utilization: Dict[str, float]
    error_rate: float
    success_rate: float

class PerformanceBenchmarker:
    """
    Comprehensive performance benchmarking and validation system.
    
    Provides tools for measuring model accuracy, system performance,
    resource utilization, and scalability characteristics.
    """
    
    def __init__(self, config: BenchmarkConfig = None, results_db: str = "benchmark_results.db"):
        self.config = config or BenchmarkConfig()
        self.results_db = results_db
        self.benchmark_results: List[SystemBenchmark] = []
        self.accuracy_results: List[AccuracyBenchmark] = []
        
        self._setup_results_database()
        logger.info(f"PerformanceBenchmarker initialized with config: {self.config}")
    
    def _setup_results_database(self):
        """Initialize database for storing benchmark results"""
        conn = sqlite3.connect(self.results_db)
        cursor = conn.cursor()
        
        cursor.executescript("""
            DROP TABLE IF EXISTS benchmark_runs;
            DROP TABLE IF EXISTS accuracy_benchmarks;
            DROP TABLE IF EXISTS performance_metrics;
            DROP TABLE IF EXISTS resource_utilization;
            
            CREATE TABLE benchmark_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT UNIQUE NOT NULL,
                component_name TEXT NOT NULL,
                benchmark_type TEXT NOT NULL,
                start_time DATETIME NOT NULL,
                end_time DATETIME,
                status TEXT CHECK(status IN ('RUNNING', 'COMPLETED', 'FAILED')),
                config_json TEXT,
                results_json TEXT
            );
            
            CREATE TABLE accuracy_benchmarks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                model_name TEXT NOT NULL,
                dataset_name TEXT NOT NULL,
                accuracy_score REAL,
                mape_score REAL,
                rmse_score REAL,
                mae_score REAL,
                prediction_count INTEGER,
                benchmark_date DATETIME NOT NULL,
                FOREIGN KEY (run_id) REFERENCES benchmark_runs(run_id)
            );
            
            CREATE TABLE performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL,
                metric_unit TEXT,
                measurement_time DATETIME NOT NULL,
                FOREIGN KEY (run_id) REFERENCES benchmark_runs(run_id)
            );
            
            CREATE TABLE resource_utilization (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                resource_type TEXT NOT NULL,
                utilization_percent REAL,
                absolute_value REAL,
                measurement_time DATETIME NOT NULL,
                FOREIGN KEY (run_id) REFERENCES benchmark_runs(run_id)
            );
        """)
        
        conn.commit()
        conn.close()
        logger.info("Benchmark results database initialized")
    
    def benchmark_model_accuracy(self, model_func: Callable, test_data: pd.DataFrame, 
                                model_name: str, dataset_name: str) -> AccuracyBenchmark:
        """Benchmark model accuracy against test dataset"""
        logger.info(f"Benchmarking accuracy for {model_name} on {dataset_name}")
        
        start_time = time.time()
        
        # Generate predictions
        predictions = []
        actual_values = []
        prediction_errors = []
        
        for _, row in test_data.iterrows():
            try:
                # Extract features and target
                features = row.drop(['target', 'date'] if 'date' in row else ['target']).to_dict()
                actual = row['target']
                
                # Make prediction
                predicted = model_func(features)
                
                predictions.append(predicted)
                actual_values.append(actual)
                
                # Calculate error
                error = abs(predicted - actual) / actual if actual != 0 else abs(predicted)
                prediction_errors.append(error)
                
            except Exception as e:
                logger.warning(f"Prediction failed for row: {e}")
                continue
        
        execution_time = time.time() - start_time
        
        # Calculate accuracy metrics
        predictions = np.array(predictions)
        actual_values = np.array(actual_values)
        
        # Mean Absolute Percentage Error (MAPE)
        mape = np.mean(np.abs((actual_values - predictions) / actual_values)) * 100
        
        # Root Mean Square Error (RMSE)
        rmse = np.sqrt(np.mean((actual_values - predictions) ** 2))
        
        # Mean Absolute Error (MAE)
        mae = np.mean(np.abs(actual_values - predictions))
        
        # Accuracy Score (1 - MAPE/100)
        accuracy = max(0, 1 - mape / 100)
        
        # Create performance metrics
        performance_metrics = PerformanceMetrics(
            accuracy_score=accuracy,
            mape_score=mape,
            rmse_score=rmse,
            mae_score=mae,
            execution_time=execution_time,
            memory_usage=0,  # Will be filled by memory profiling
            cpu_usage=0,     # Will be filled by CPU profiling
            throughput=len(predictions) / execution_time,
            latency_p50=0,  # Not applicable for batch processing
            latency_p95=0,
            latency_p99=0
        )
        
        # Calculate feature importance (simplified)
        feature_importance = self._calculate_feature_importance(test_data, predictions)
        
        benchmark = AccuracyBenchmark(
            model_name=model_name,
            dataset_name=dataset_name,
            accuracy_metrics=performance_metrics,
            prediction_errors=prediction_errors,
            feature_importance=feature_importance
        )
        
        self.accuracy_results.append(benchmark)
        self._save_accuracy_benchmark(benchmark)
        
        logger.info(f"Accuracy benchmark completed: {accuracy:.3f} accuracy, {mape:.2f}% MAPE")
        return benchmark
    
    def _calculate_feature_importance(self, data: pd.DataFrame, predictions: np.ndarray) -> Dict[str, float]:
        """Calculate simplified feature importance scores"""
        feature_importance = {}
        
        # Get numeric features only
        numeric_features = data.select_dtypes(include=[np.number]).columns
        numeric_features = [col for col in numeric_features if col not in ['target', 'date']]
        
        for feature in numeric_features:
            if feature in data.columns:
                # Calculate correlation with predictions as proxy for importance
                correlation = np.corrcoef(data[feature].fillna(0), predictions)[0, 1]
                feature_importance[feature] = abs(correlation) if not np.isnan(correlation) else 0.0
        
        # Normalize importance scores
        total_importance = sum(feature_importance.values())
        if total_importance > 0:
            feature_importance = {k: v / total_importance for k, v in feature_importance.items()}
        
        return feature_importance
    
    async def benchmark_system_performance(self, component_func: Callable, 
                                         component_name: str, 
                                         test_inputs: List[Any]) -> SystemBenchmark:
        """Benchmark system performance under load"""
        logger.info(f"Benchmarking system performance for {component_name}")
        
        run_id = f"{component_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Start resource monitoring
        if self.config.memory_profiling:
            tracemalloc.start()
        
        start_time = time.time()
        initial_cpu = psutil.cpu_percent()
        initial_memory = psutil.virtual_memory().percent
        
        # Warmup phase
        logger.info("Running warmup requests...")
        await self._run_warmup_requests(component_func, test_inputs[:self.config.warmup_requests])
        
        # Main benchmark phase
        logger.info(f"Running main benchmark with {self.config.concurrent_requests} concurrent requests")
        
        latencies = []
        errors = 0
        successes = 0
        
        # Run concurrent requests
        with ThreadPoolExecutor(max_workers=self.config.concurrent_requests) as executor:
            futures = []
            
            # Submit requests
            for i in range(len(test_inputs)):
                if i >= len(test_inputs):
                    break
                
                future = executor.submit(self._execute_with_timing, component_func, test_inputs[i])
                futures.append(future)
            
            # Collect results
            for future in as_completed(futures):
                try:
                    latency, success = future.result()
                    latencies.append(latency)
                    if success:
                        successes += 1
                    else:
                        errors += 1
                except Exception as e:
                    logger.warning(f"Request failed: {e}")
                    errors += 1
        
        end_time = time.time()
        total_duration = end_time - start_time
        
        # Calculate performance metrics
        final_cpu = psutil.cpu_percent()
        final_memory = psutil.virtual_memory().percent
        
        # Memory profiling results
        memory_usage = 0
        if self.config.memory_profiling and tracemalloc.is_tracing():
            current, peak = tracemalloc.get_traced_memory()
            memory_usage = peak / 1024 / 1024  # Convert to MB
            tracemalloc.stop()
        
        # Calculate latency percentiles
        latencies.sort()
        latency_p50 = np.percentile(latencies, 50) if latencies else 0
        latency_p95 = np.percentile(latencies, 95) if latencies else 0
        latency_p99 = np.percentile(latencies, 99) if latencies else 0
        
        # Calculate throughput
        total_requests = successes + errors
        throughput = total_requests / total_duration if total_duration > 0 else 0
        
        performance_metrics = PerformanceMetrics(
            accuracy_score=0,  # Not applicable for system benchmarks
            mape_score=0,
            rmse_score=0,
            mae_score=0,
            execution_time=total_duration,
            memory_usage=memory_usage,
            cpu_usage=final_cpu - initial_cpu,
            throughput=throughput,
            latency_p50=latency_p50 * 1000,  # Convert to milliseconds
            latency_p95=latency_p95 * 1000,
            latency_p99=latency_p99 * 1000
        )
        
        resource_utilization = {
            'cpu_usage_delta': final_cpu - initial_cpu,
            'memory_usage_delta': final_memory - initial_memory,
            'peak_memory_mb': memory_usage
        }
        
        error_rate = errors / total_requests if total_requests > 0 else 0
        success_rate = successes / total_requests if total_requests > 0 else 0
        
        benchmark = SystemBenchmark(
            component_name=component_name,
            benchmark_config=self.config,
            performance_metrics=performance_metrics,
            resource_utilization=resource_utilization,
            error_rate=error_rate,
            success_rate=success_rate
        )
        
        self.benchmark_results.append(benchmark)
        self._save_system_benchmark(benchmark, run_id)
        
        logger.info(f"System benchmark completed: {throughput:.1f} req/s, {latency_p95:.1f}ms p95 latency")
        return benchmark
    
    async def _run_warmup_requests(self, component_func: Callable, warmup_inputs: List[Any]):
        """Run warmup requests to stabilize performance"""
        for input_data in warmup_inputs:
            try:
                await asyncio.get_event_loop().run_in_executor(None, component_func, input_data)
            except Exception as e:
                logger.debug(f"Warmup request failed: {e}")
    
    def _execute_with_timing(self, func: Callable, input_data: Any) -> Tuple[float, bool]:
        """Execute function with timing measurement"""
        start_time = time.time()
        success = False
        
        try:
            result = func(input_data)
            success = result is not None
        except Exception as e:
            logger.debug(f"Function execution failed: {e}")
        
        end_time = time.time()
        latency = end_time - start_time
        
        return latency, success
    
    def benchmark_drift_detection_accuracy(self, drift_detector_func: Callable, 
                                         test_scenarios: List[Dict]) -> Dict[str, float]:
        """Benchmark drift detection accuracy with known drift scenarios"""
        logger.info("Benchmarking drift detection accuracy")
        
        true_positives = 0
        false_positives = 0
        true_negatives = 0
        false_negatives = 0
        
        for scenario in test_scenarios:
            has_drift = scenario.get('has_drift', False)
            data = scenario.get('data')
            
            try:
                detected_drift = drift_detector_func(data)
                
                if has_drift and detected_drift:
                    true_positives += 1
                elif has_drift and not detected_drift:
                    false_negatives += 1
                elif not has_drift and detected_drift:
                    false_positives += 1
                else:
                    true_negatives += 1
                    
            except Exception as e:
                logger.warning(f"Drift detection failed for scenario: {e}")
                false_negatives += 1
        
        # Calculate metrics
        total = len(test_scenarios)
        accuracy = (true_positives + true_negatives) / total if total > 0 else 0
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'true_negatives': true_negatives,
            'false_negatives': false_negatives
        }
        
        logger.info(f"Drift detection accuracy: {accuracy:.3f}, F1: {f1_score:.3f}")
        return results
    
    def benchmark_end_to_end_latency(self, workflow_func: Callable, 
                                   test_inputs: List[Any]) -> Dict[str, float]:
        """Benchmark end-to-end workflow latency"""
        logger.info("Benchmarking end-to-end workflow latency")
        
        latencies = []
        
        for input_data in test_inputs:
            start_time = time.time()
            
            try:
                result = workflow_func(input_data)
                end_time = time.time()
                
                if result is not None:
                    latency = end_time - start_time
                    latencies.append(latency)
                    
            except Exception as e:
                logger.warning(f"Workflow execution failed: {e}")
        
        if not latencies:
            return {'error': 'No successful executions'}
        
        results = {
            'mean_latency': statistics.mean(latencies),
            'median_latency': statistics.median(latencies),
            'p95_latency': np.percentile(latencies, 95),
            'p99_latency': np.percentile(latencies, 99),
            'min_latency': min(latencies),
            'max_latency': max(latencies),
            'std_latency': statistics.stdev(latencies) if len(latencies) > 1 else 0
        }
        
        logger.info(f"End-to-end latency - Mean: {results['mean_latency']:.3f}s, P95: {results['p95_latency']:.3f}s")
        return results
    
    def validate_performance_requirements(self) -> Dict[str, bool]:
        """Validate system performance against requirements"""
        logger.info("Validating performance requirements")
        
        validation_results = {}
        
        for benchmark in self.benchmark_results:
            component = benchmark.component_name
            metrics = benchmark.performance_metrics
            
            # Validate latency requirements
            latency_ok = metrics.latency_p95 <= self.config.latency_threshold_ms
            validation_results[f"{component}_latency"] = latency_ok
            
            # Validate throughput requirements
            throughput_ok = metrics.throughput >= self.config.throughput_threshold
            validation_results[f"{component}_throughput"] = throughput_ok
            
            # Validate error rate
            error_rate_ok = benchmark.error_rate <= 0.01  # 1% error rate threshold
            validation_results[f"{component}_error_rate"] = error_rate_ok
        
        for accuracy_benchmark in self.accuracy_results:
            model = accuracy_benchmark.model_name
            accuracy_ok = accuracy_benchmark.accuracy_metrics.accuracy_score >= self.config.accuracy_threshold
            validation_results[f"{model}_accuracy"] = accuracy_ok
        
        # Overall validation
        all_passed = all(validation_results.values())
        validation_results['overall_validation'] = all_passed
        
        logger.info(f"Performance validation: {'PASSED' if all_passed else 'FAILED'}")
        return validation_results
    
    def generate_performance_report(self, output_file: str = "performance_report.json") -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        logger.info("Generating performance report")
        
        report = {
            'benchmark_summary': {
                'total_system_benchmarks': len(self.benchmark_results),
                'total_accuracy_benchmarks': len(self.accuracy_results),
                'benchmark_date': datetime.now().isoformat(),
                'config': asdict(self.config)
            },
            'system_performance': [],
            'accuracy_results': [],
            'validation_results': self.validate_performance_requirements(),
            'recommendations': self._generate_recommendations()
        }
        
        # Add system performance results
        for benchmark in self.benchmark_results:
            report['system_performance'].append({
                'component': benchmark.component_name,
                'metrics': asdict(benchmark.performance_metrics),
                'resource_utilization': benchmark.resource_utilization,
                'error_rate': benchmark.error_rate,
                'success_rate': benchmark.success_rate
            })
        
        # Add accuracy results
        for accuracy in self.accuracy_results:
            report['accuracy_results'].append({
                'model': accuracy.model_name,
                'dataset': accuracy.dataset_name,
                'metrics': asdict(accuracy.accuracy_metrics),
                'feature_importance': accuracy.feature_importance,
                'error_statistics': {
                    'mean_error': statistics.mean(accuracy.prediction_errors),
                    'median_error': statistics.median(accuracy.prediction_errors),
                    'max_error': max(accuracy.prediction_errors),
                    'error_std': statistics.stdev(accuracy.prediction_errors) if len(accuracy.prediction_errors) > 1 else 0
                }
            })
        
        # Save report
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Performance report saved to {output_file}")
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate performance improvement recommendations"""
        recommendations = []
        
        for benchmark in self.benchmark_results:
            metrics = benchmark.performance_metrics
            
            if metrics.latency_p95 > self.config.latency_threshold_ms:
                recommendations.append(
                    f"High latency detected in {benchmark.component_name} "
                    f"({metrics.latency_p95:.1f}ms). Consider optimization or caching."
                )
            
            if metrics.throughput < self.config.throughput_threshold:
                recommendations.append(
                    f"Low throughput in {benchmark.component_name} "
                    f"({metrics.throughput:.1f} req/s). Consider scaling or optimization."
                )
            
            if benchmark.error_rate > 0.01:
                recommendations.append(
                    f"High error rate in {benchmark.component_name} "
                    f"({benchmark.error_rate:.2%}). Investigate error handling."
                )
        
        for accuracy in self.accuracy_results:
            if accuracy.accuracy_metrics.accuracy_score < self.config.accuracy_threshold:
                recommendations.append(
                    f"Low accuracy in {accuracy.model_name} "
                    f"({accuracy.accuracy_metrics.accuracy_score:.3f}). Consider model retraining."
                )
        
        if not recommendations:
            recommendations.append("All performance metrics meet requirements.")
        
        return recommendations
    
    def _save_accuracy_benchmark(self, benchmark: AccuracyBenchmark):
        """Save accuracy benchmark to database"""
        conn = sqlite3.connect(self.results_db)
        cursor = conn.cursor()
        
        run_id = f"{benchmark.model_name}_{benchmark.dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        cursor.execute("""
            INSERT INTO accuracy_benchmarks 
            (run_id, model_name, dataset_name, accuracy_score, mape_score, rmse_score, mae_score, 
             prediction_count, benchmark_date)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            run_id, benchmark.model_name, benchmark.dataset_name,
            benchmark.accuracy_metrics.accuracy_score, benchmark.accuracy_metrics.mape_score,
            benchmark.accuracy_metrics.rmse_score, benchmark.accuracy_metrics.mae_score,
            len(benchmark.prediction_errors), datetime.now()
        ))
        
        conn.commit()
        conn.close()
    
    def _save_system_benchmark(self, benchmark: SystemBenchmark, run_id: str):
        """Save system benchmark to database"""
        conn = sqlite3.connect(self.results_db)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO benchmark_runs 
            (run_id, component_name, benchmark_type, start_time, end_time, status, config_json, results_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            run_id, benchmark.component_name, 'system_performance',
            datetime.now(), datetime.now(), 'COMPLETED',
            json.dumps(asdict(benchmark.benchmark_config)),
            json.dumps(asdict(benchmark.performance_metrics))
        ))
        
        conn.commit()
        conn.close()
    
    def cleanup(self):
        """Clean up benchmark resources"""
        if Path(self.results_db).exists():
            Path(self.results_db).unlink()
        logger.info("Benchmark cleanup completed")