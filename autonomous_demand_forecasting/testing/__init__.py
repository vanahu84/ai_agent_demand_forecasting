"""
Comprehensive Testing and Validation Framework for Autonomous Demand Forecasting

This module provides retail simulation, synthetic data generation, performance benchmarking,
and comprehensive integration testing capabilities for the autonomous demand forecasting system.
"""

__version__ = "1.0.0"
__author__ = "Autonomous Demand Forecasting Team"

from .retail_simulator import RetailSimulator
from .synthetic_data_generator import SyntheticDataGenerator
from .performance_benchmarker import PerformanceBenchmarker
from .integration_test_suite import IntegrationTestSuite

__all__ = [
    'RetailSimulator',
    'SyntheticDataGenerator', 
    'PerformanceBenchmarker',
    'IntegrationTestSuite'
]