"""
Synthetic Data Generation with Realistic Seasonal Patterns

This module provides comprehensive synthetic data generation capabilities
for testing the autonomous demand forecasting system with realistic patterns.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import random
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class SeasonalPattern:
    """Defines seasonal pattern parameters"""
    name: str
    amplitude: float
    phase_shift: int  # Days
    frequency: float  # Cycles per year
    noise_level: float
    trend_slope: float

@dataclass
class DataGenerationConfig:
    """Configuration for synthetic data generation"""
    start_date: datetime
    end_date: datetime
    num_products: int = 50
    num_stores: int = 5
    base_demand_range: Tuple[int, int] = (10, 100)
    noise_level: float = 0.1
    trend_probability: float = 0.3
    seasonal_probability: float = 0.8
    promotion_probability: float = 0.15
    stockout_probability: float = 0.05
    random_seed: Optional[int] = 42

class SyntheticDataGenerator:
    """
    Advanced synthetic data generator for retail demand forecasting testing.
    
    Generates realistic time series data with seasonal patterns, trends, promotions,
    stockouts, and various market conditions for comprehensive system testing.
    """
    
    def __init__(self, config: DataGenerationConfig):
        self.config = config
        self.seasonal_patterns = self._define_seasonal_patterns()
        
        if config.random_seed:
            random.seed(config.random_seed)
            np.random.seed(config.random_seed)
        
        logger.info(f"SyntheticDataGenerator initialized with config: {config}")
    
    def _define_seasonal_patterns(self) -> Dict[str, SeasonalPattern]:
        """Define various seasonal patterns for different product categories"""
        return {
            'holiday': SeasonalPattern(
                name='holiday',
                amplitude=0.5,
                phase_shift=330,  # Peak in late November
                frequency=1.0,
                noise_level=0.15,
                trend_slope=0.02
            ),
            'summer': SeasonalPattern(
                name='summer',
                amplitude=0.3,
                phase_shift=180,  # Peak in June
                frequency=1.0,
                noise_level=0.1,
                trend_slope=0.01
            ),
            'winter': SeasonalPattern(
                name='winter',
                amplitude=0.4,
                phase_shift=0,  # Peak in December/January
                frequency=1.0,
                noise_level=0.12,
                trend_slope=-0.005
            ),
            'back_to_school': SeasonalPattern(
                name='back_to_school',
                amplitude=0.6,
                phase_shift=240,  # Peak in late August
                frequency=1.0,
                noise_level=0.2,
                trend_slope=0.015
            ),
            'steady': SeasonalPattern(
                name='steady',
                amplitude=0.05,
                phase_shift=0,
                frequency=1.0,
                noise_level=0.08,
                trend_slope=0.005
            ),
            'weekly': SeasonalPattern(
                name='weekly',
                amplitude=0.2,
                phase_shift=5,  # Peak on Friday
                frequency=52.0,  # Weekly pattern
                noise_level=0.15,
                trend_slope=0.0
            )
        }
    
    def generate_demand_time_series(self, product_id: str, store_id: str, 
                                  base_demand: int, pattern_name: str) -> pd.DataFrame:
        """Generate realistic demand time series for a product-store combination"""
        
        date_range = pd.date_range(
            start=self.config.start_date,
            end=self.config.end_date,
            freq='D'
        )
        
        pattern = self.seasonal_patterns.get(pattern_name, self.seasonal_patterns['steady'])
        
        # Generate base time series
        days = len(date_range)
        time_index = np.arange(days)
        
        # Seasonal component
        seasonal_component = pattern.amplitude * np.sin(
            2 * np.pi * pattern.frequency * (time_index + pattern.phase_shift) / 365.25
        )
        
        # Trend component
        trend_component = pattern.trend_slope * time_index / 365.25
        
        # Weekly pattern overlay
        weekly_pattern = self.seasonal_patterns['weekly']
        weekly_component = 0.1 * weekly_pattern.amplitude * np.sin(
            2 * np.pi * weekly_pattern.frequency * (time_index + weekly_pattern.phase_shift) / 365.25
        )
        
        # Noise component
        noise_component = np.random.normal(0, pattern.noise_level, days)
        
        # Combine components
        demand_multiplier = 1 + seasonal_component + trend_component + weekly_component + noise_component
        demand_multiplier = np.maximum(demand_multiplier, 0.1)  # Ensure positive demand
        
        raw_demand = base_demand * demand_multiplier
        
        # Add promotions
        promotion_demand = self._add_promotion_effects(raw_demand, date_range)
        
        # Add stockouts
        final_demand = self._add_stockout_effects(promotion_demand, date_range)
        
        # Create DataFrame
        df = pd.DataFrame({
            'date': date_range,
            'product_id': product_id,
            'store_id': store_id,
            'base_demand': base_demand,
            'seasonal_multiplier': 1 + seasonal_component,
            'trend_multiplier': 1 + trend_component,
            'weekly_multiplier': 1 + weekly_component,
            'noise_multiplier': 1 + noise_component,
            'raw_demand': raw_demand,
            'promotion_applied': False,
            'stockout_applied': False,
            'final_demand': final_demand.astype(int),
            'pattern_name': pattern_name
        })
        
        return df
    
    def _add_promotion_effects(self, demand: np.ndarray, date_range: pd.DatetimeIndex) -> np.ndarray:
        """Add realistic promotion effects to demand time series"""
        promotion_demand = demand.copy()
        
        # Generate random promotion periods
        num_promotions = int(len(date_range) * self.config.promotion_probability / 30)  # Average 30-day promotions
        
        for _ in range(num_promotions):
            # Random promotion start
            start_idx = random.randint(0, len(date_range) - 14)  # At least 14 days before end
            duration = random.randint(3, 21)  # 3-21 day promotions
            end_idx = min(start_idx + duration, len(date_range))
            
            # Promotion effect (increase demand)
            promotion_multiplier = random.uniform(1.2, 2.5)
            promotion_demand[start_idx:end_idx] *= promotion_multiplier
            
            # Post-promotion dip (cannibalization effect)
            post_promotion_duration = min(duration // 2, len(date_range) - end_idx)
            if post_promotion_duration > 0:
                dip_multiplier = random.uniform(0.7, 0.9)
                promotion_demand[end_idx:end_idx + post_promotion_duration] *= dip_multiplier
        
        return promotion_demand
    
    def _add_stockout_effects(self, demand: np.ndarray, date_range: pd.DatetimeIndex) -> np.ndarray:
        """Add realistic stockout effects to demand time series"""
        stockout_demand = demand.copy()
        
        # Generate random stockout periods
        num_stockouts = int(len(date_range) * self.config.stockout_probability / 7)  # Average 7-day stockouts
        
        for _ in range(num_stockouts):
            # Random stockout start
            start_idx = random.randint(0, len(date_range) - 3)  # At least 3 days before end
            duration = random.randint(1, 7)  # 1-7 day stockouts
            end_idx = min(start_idx + duration, len(date_range))
            
            # Stockout effect (zero demand)
            stockout_demand[start_idx:end_idx] = 0
            
            # Post-stockout surge (pent-up demand)
            surge_duration = min(duration, len(date_range) - end_idx)
            if surge_duration > 0:
                surge_multiplier = random.uniform(1.3, 1.8)
                stockout_demand[end_idx:end_idx + surge_duration] *= surge_multiplier
        
        return stockout_demand
    
    def generate_external_factors(self) -> pd.DataFrame:
        """Generate external factors that influence demand (weather, events, etc.)"""
        date_range = pd.date_range(
            start=self.config.start_date,
            end=self.config.end_date,
            freq='D'
        )
        
        days = len(date_range)
        
        # Weather factors
        temperature = 20 + 15 * np.sin(2 * np.pi * np.arange(days) / 365.25) + np.random.normal(0, 5, days)
        precipitation = np.maximum(0, np.random.exponential(2, days))
        
        # Economic factors
        economic_index = 100 + np.cumsum(np.random.normal(0, 0.5, days))
        
        # Special events (holidays, sales events, etc.)
        special_events = np.zeros(days)
        holiday_dates = self._get_holiday_dates(date_range)
        for holiday_idx in holiday_dates:
            if 0 <= holiday_idx < days:
                special_events[holiday_idx] = 1
        
        # Marketing spend (affects demand)
        marketing_spend = np.random.lognormal(8, 1, days)  # Log-normal distribution
        
        df = pd.DataFrame({
            'date': date_range,
            'temperature': temperature,
            'precipitation': precipitation,
            'economic_index': economic_index,
            'special_event': special_events,
            'marketing_spend': marketing_spend,
            'day_of_week': date_range.dayofweek,
            'month': date_range.month,
            'quarter': date_range.quarter,
            'is_weekend': date_range.dayofweek >= 5,
            'is_month_end': date_range.is_month_end,
            'is_quarter_end': date_range.is_quarter_end
        })
        
        return df
    
    def _get_holiday_dates(self, date_range: pd.DatetimeIndex) -> List[int]:
        """Get indices of major holidays in the date range"""
        holiday_indices = []
        
        for i, date in enumerate(date_range):
            # New Year's Day
            if date.month == 1 and date.day == 1:
                holiday_indices.append(i)
            
            # Valentine's Day
            elif date.month == 2 and date.day == 14:
                holiday_indices.append(i)
            
            # Easter (approximate - first Sunday after March 21)
            elif date.month == 3 and date.day > 21 and date.dayofweek == 6:
                holiday_indices.append(i)
            
            # Mother's Day (second Sunday in May)
            elif date.month == 5 and date.dayofweek == 6 and 8 <= date.day <= 14:
                holiday_indices.append(i)
            
            # Independence Day
            elif date.month == 7 and date.day == 4:
                holiday_indices.append(i)
            
            # Labor Day (first Monday in September)
            elif date.month == 9 and date.dayofweek == 0 and date.day <= 7:
                holiday_indices.append(i)
            
            # Halloween
            elif date.month == 10 and date.day == 31:
                holiday_indices.append(i)
            
            # Thanksgiving (fourth Thursday in November)
            elif date.month == 11 and date.dayofweek == 3 and 22 <= date.day <= 28:
                holiday_indices.append(i)
            
            # Black Friday
            elif date.month == 11 and date.dayofweek == 4 and 23 <= date.day <= 29:
                holiday_indices.append(i)
            
            # Christmas
            elif date.month == 12 and date.day == 25:
                holiday_indices.append(i)
        
        return holiday_indices
    
    def generate_price_data(self, product_ids: List[str]) -> pd.DataFrame:
        """Generate realistic price data with occasional price changes"""
        date_range = pd.date_range(
            start=self.config.start_date,
            end=self.config.end_date,
            freq='D'
        )
        
        price_data = []
        
        for product_id in product_ids:
            base_price = random.uniform(10, 500)
            current_price = base_price
            
            for date in date_range:
                # Occasional price changes (5% chance per day)
                if random.random() < 0.05:
                    price_change = random.uniform(-0.2, 0.2)  # ±20% change
                    current_price = max(base_price * 0.5, current_price * (1 + price_change))
                
                price_data.append({
                    'date': date,
                    'product_id': product_id,
                    'price': round(current_price, 2),
                    'base_price': round(base_price, 2),
                    'price_change_pct': round((current_price - base_price) / base_price * 100, 2)
                })
        
        return pd.DataFrame(price_data)
    
    def generate_competitor_data(self, product_ids: List[str]) -> pd.DataFrame:
        """Generate competitor pricing and activity data"""
        date_range = pd.date_range(
            start=self.config.start_date,
            end=self.config.end_date,
            freq='W'  # Weekly competitor data
        )
        
        competitor_data = []
        competitors = ['CompetitorA', 'CompetitorB', 'CompetitorC']
        
        for product_id in product_ids:
            base_competitor_price = random.uniform(8, 520)  # Slightly different from our prices
            
            for date in date_range:
                for competitor in competitors:
                    # Competitor price with some variation
                    price_variation = random.uniform(-0.15, 0.15)
                    competitor_price = base_competitor_price * (1 + price_variation)
                    
                    # Competitor promotion activity
                    has_promotion = random.random() < 0.2  # 20% chance of promotion
                    promotion_discount = random.uniform(0.1, 0.3) if has_promotion else 0.0
                    
                    competitor_data.append({
                        'date': date,
                        'product_id': product_id,
                        'competitor': competitor,
                        'price': round(competitor_price, 2),
                        'has_promotion': has_promotion,
                        'promotion_discount': round(promotion_discount, 2),
                        'market_share': random.uniform(0.1, 0.4)
                    })
        
        return pd.DataFrame(competitor_data)
    
    def generate_inventory_data(self, product_ids: List[str], store_ids: List[str]) -> pd.DataFrame:
        """Generate realistic inventory level data"""
        date_range = pd.date_range(
            start=self.config.start_date,
            end=self.config.end_date,
            freq='D'
        )
        
        inventory_data = []
        
        for product_id in product_ids:
            for store_id in store_ids:
                # Initial inventory levels
                max_stock = random.randint(100, 1000)
                reorder_point = int(max_stock * 0.2)
                current_stock = random.randint(reorder_point, max_stock)
                
                for date in date_range:
                    # Simulate daily inventory changes
                    daily_sales = random.randint(0, 20)
                    daily_receipts = 0
                    
                    # Reorder when below reorder point
                    if current_stock <= reorder_point:
                        daily_receipts = random.randint(max_stock // 2, max_stock)
                    
                    current_stock = max(0, current_stock - daily_sales + daily_receipts)
                    
                    inventory_data.append({
                        'date': date,
                        'product_id': product_id,
                        'store_id': store_id,
                        'current_stock': current_stock,
                        'max_stock': max_stock,
                        'reorder_point': reorder_point,
                        'daily_sales': daily_sales,
                        'daily_receipts': daily_receipts,
                        'stock_out': current_stock == 0,
                        'stock_level_pct': round(current_stock / max_stock * 100, 1)
                    })
        
        return pd.DataFrame(inventory_data)
    
    def generate_customer_behavior_data(self) -> pd.DataFrame:
        """Generate customer behavior and segmentation data"""
        date_range = pd.date_range(
            start=self.config.start_date,
            end=self.config.end_date,
            freq='W'  # Weekly customer behavior data
        )
        
        segments = ['Premium', 'Regular', 'Budget', 'Occasional']
        behavior_data = []
        
        for date in date_range:
            for segment in segments:
                # Segment-specific behavior patterns
                if segment == 'Premium':
                    avg_basket_size = random.uniform(150, 300)
                    price_sensitivity = random.uniform(0.1, 0.3)
                    brand_loyalty = random.uniform(0.7, 0.9)
                elif segment == 'Regular':
                    avg_basket_size = random.uniform(75, 150)
                    price_sensitivity = random.uniform(0.3, 0.6)
                    brand_loyalty = random.uniform(0.4, 0.7)
                elif segment == 'Budget':
                    avg_basket_size = random.uniform(30, 75)
                    price_sensitivity = random.uniform(0.6, 0.9)
                    brand_loyalty = random.uniform(0.2, 0.4)
                else:  # Occasional
                    avg_basket_size = random.uniform(20, 60)
                    price_sensitivity = random.uniform(0.4, 0.8)
                    brand_loyalty = random.uniform(0.1, 0.3)
                
                behavior_data.append({
                    'date': date,
                    'customer_segment': segment,
                    'avg_basket_size': round(avg_basket_size, 2),
                    'price_sensitivity': round(price_sensitivity, 2),
                    'brand_loyalty': round(brand_loyalty, 2),
                    'purchase_frequency': random.uniform(0.5, 3.0),
                    'digital_engagement': random.uniform(0.1, 1.0),
                    'promotion_responsiveness': random.uniform(0.2, 0.8)
                })
        
        return pd.DataFrame(behavior_data)
    
    def generate_comprehensive_dataset(self, output_dir: str = "synthetic_data") -> Dict[str, pd.DataFrame]:
        """Generate comprehensive synthetic dataset for testing"""
        logger.info("Generating comprehensive synthetic dataset")
        
        # Generate product and store IDs
        product_ids = [f"PROD_{i:04d}" for i in range(self.config.num_products)]
        store_ids = [f"STORE_{i:03d}" for i in range(self.config.num_stores)]
        
        # Generate all data components
        datasets = {}
        
        # Demand time series for each product-store combination
        demand_data = []
        patterns = list(self.seasonal_patterns.keys())
        
        for product_id in product_ids:
            pattern = random.choice(patterns)
            base_demand = random.randint(*self.config.base_demand_range)
            
            for store_id in store_ids:
                store_multiplier = random.uniform(0.5, 2.0)  # Store-specific demand variation
                adjusted_demand = int(base_demand * store_multiplier)
                
                demand_ts = self.generate_demand_time_series(
                    product_id, store_id, adjusted_demand, pattern
                )
                demand_data.append(demand_ts)
        
        datasets['demand'] = pd.concat(demand_data, ignore_index=True)
        
        # External factors
        datasets['external_factors'] = self.generate_external_factors()
        
        # Price data
        datasets['prices'] = self.generate_price_data(product_ids)
        
        # Competitor data
        datasets['competitors'] = self.generate_competitor_data(product_ids)
        
        # Inventory data
        datasets['inventory'] = self.generate_inventory_data(product_ids, store_ids)
        
        # Customer behavior data
        datasets['customer_behavior'] = self.generate_customer_behavior_data()
        
        # Save datasets to files
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        for name, df in datasets.items():
            df.to_csv(output_path / f"{name}.csv", index=False)
            logger.info(f"Saved {name} dataset with {len(df)} records")
        
        # Generate metadata
        metadata = {
            'generation_config': {
                'start_date': self.config.start_date.isoformat(),
                'end_date': self.config.end_date.isoformat(),
                'num_products': self.config.num_products,
                'num_stores': self.config.num_stores,
                'random_seed': self.config.random_seed
            },
            'datasets': {
                name: {
                    'records': len(df),
                    'columns': list(df.columns),
                    'date_range': {
                        'start': df['date'].min().isoformat() if 'date' in df.columns else None,
                        'end': df['date'].max().isoformat() if 'date' in df.columns else None
                    }
                }
                for name, df in datasets.items()
            },
            'seasonal_patterns': {
                name: {
                    'amplitude': pattern.amplitude,
                    'phase_shift': pattern.phase_shift,
                    'frequency': pattern.frequency,
                    'noise_level': pattern.noise_level,
                    'trend_slope': pattern.trend_slope
                }
                for name, pattern in self.seasonal_patterns.items()
            }
        }
        
        with open(output_path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Generated comprehensive dataset with {len(datasets)} components")
        return datasets
    
    def add_data_quality_issues(self, datasets: Dict[str, pd.DataFrame], 
                              missing_rate: float = 0.02, 
                              outlier_rate: float = 0.01) -> Dict[str, pd.DataFrame]:
        """Add realistic data quality issues for testing data cleaning algorithms"""
        logger.info(f"Adding data quality issues: {missing_rate*100}% missing, {outlier_rate*100}% outliers")
        
        corrupted_datasets = {}
        
        for name, df in datasets.items():
            corrupted_df = df.copy()
            
            # Add missing values
            for col in df.select_dtypes(include=[np.number]).columns:
                if col != 'date':  # Don't corrupt date columns
                    missing_mask = np.random.random(len(df)) < missing_rate
                    corrupted_df.loc[missing_mask, col] = np.nan
            
            # Add outliers
            for col in df.select_dtypes(include=[np.number]).columns:
                if col not in ['date', 'product_id', 'store_id']:
                    outlier_mask = np.random.random(len(df)) < outlier_rate
                    if outlier_mask.any():
                        # Generate extreme values (5-10x normal range)
                        col_std = df[col].std()
                        col_mean = df[col].mean()
                        outlier_values = np.random.normal(col_mean, col_std * 5, outlier_mask.sum())
                        corrupted_df.loc[outlier_mask, col] = outlier_values
            
            corrupted_datasets[name] = corrupted_df
        
        return corrupted_datasets
    
    def generate_test_scenarios(self) -> List[Dict[str, Any]]:
        """Generate specific test scenarios for system validation"""
        scenarios = [
            {
                'name': 'holiday_surge',
                'description': 'Black Friday demand surge scenario',
                'modifications': {
                    'demand_multiplier': 3.0,
                    'date_range': ('2024-11-29', '2024-12-02'),
                    'affected_categories': ['Electronics', 'Clothing']
                }
            },
            {
                'name': 'supply_disruption',
                'description': 'Supply chain disruption causing stockouts',
                'modifications': {
                    'stockout_probability': 0.3,
                    'duration_days': 14,
                    'affected_products': 0.2  # 20% of products affected
                }
            },
            {
                'name': 'new_competitor',
                'description': 'New competitor entry affecting market share',
                'modifications': {
                    'demand_reduction': 0.15,
                    'price_pressure': 0.1,
                    'affected_categories': ['Electronics']
                }
            },
            {
                'name': 'economic_downturn',
                'description': 'Economic recession affecting customer behavior',
                'modifications': {
                    'demand_reduction': 0.25,
                    'price_sensitivity_increase': 0.3,
                    'premium_segment_impact': 0.4
                }
            },
            {
                'name': 'seasonal_shift',
                'description': 'Unexpected seasonal pattern change',
                'modifications': {
                    'pattern_shift_days': 30,
                    'amplitude_change': 0.5,
                    'affected_patterns': ['summer', 'winter']
                }
            }
        ]
        
        return scenarios