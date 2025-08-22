"""
Retail Sales and Inventory Simulation Framework

This module provides comprehensive retail environment simulation for testing
the autonomous demand forecasting system with realistic business scenarios.
"""

import asyncio
import random
import sqlite3
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)

@dataclass
class SimulationConfig:
    """Configuration for retail simulation parameters"""
    num_products: int = 100
    num_stores: int = 10
    num_customers: int = 1000
    simulation_days: int = 365
    base_demand_min: int = 10
    base_demand_max: int = 100
    seasonal_amplitude: float = 0.3
    promotion_frequency: float = 0.1
    stockout_probability: float = 0.05
    price_elasticity: float = -0.5
    random_seed: Optional[int] = 42

@dataclass
class Product:
    """Product entity for simulation"""
    id: str
    name: str
    category: str
    subcategory: str
    brand: str
    unit_price: float
    cost: float
    base_demand: int
    seasonality_pattern: str
    price_elasticity: float

@dataclass
class Store:
    """Store entity for simulation"""
    id: str
    name: str
    location: str
    region: str
    store_type: str
    square_footage: int
    customer_traffic_multiplier: float

@dataclass
class Customer:
    """Customer entity for simulation"""
    id: str
    segment: str
    location_preference: str
    price_sensitivity: float
    brand_loyalty: Dict[str, float]
    purchase_frequency: float

@dataclass
class SalesTransaction:
    """Sales transaction for simulation"""
    transaction_id: str
    product_id: str
    store_id: str
    customer_id: str
    quantity: int
    unit_price: float
    total_amount: float
    transaction_date: datetime
    promotion_applied: bool
    discount_amount: float

@dataclass
class InventoryRecord:
    """Inventory record for simulation"""
    product_id: str
    store_id: str
    current_stock: int
    reserved_stock: int
    available_stock: int
    reorder_point: int
    max_stock_level: int
    last_updated: datetime

class RetailSimulator:
    """
    Comprehensive retail environment simulator for testing autonomous demand forecasting.
    
    Generates realistic sales transactions, inventory movements, customer behavior,
    and seasonal patterns to validate system performance under various scenarios.
    """
    
    def __init__(self, config: SimulationConfig = None, db_path: str = "test_retail_simulation.db"):
        self.config = config or SimulationConfig()
        self.db_path = db_path
        self.products: List[Product] = []
        self.stores: List[Store] = []
        self.customers: List[Customer] = []
        self.sales_transactions: List[SalesTransaction] = []
        self.inventory_records: List[InventoryRecord] = []
        
        if self.config.random_seed:
            random.seed(self.config.random_seed)
            np.random.seed(self.config.random_seed)
        
        self._setup_database()
        logger.info(f"RetailSimulator initialized with config: {self.config}")
    
    def _setup_database(self):
        """Initialize simulation database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create simulation tables
        cursor.executescript("""
            DROP TABLE IF EXISTS sim_products;
            DROP TABLE IF EXISTS sim_stores;
            DROP TABLE IF EXISTS sim_customers;
            DROP TABLE IF EXISTS sim_sales_transactions;
            DROP TABLE IF EXISTS sim_inventory_records;
            DROP TABLE IF EXISTS sim_seasonal_patterns;
            DROP TABLE IF EXISTS sim_promotions;
            
            CREATE TABLE sim_products (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                category TEXT NOT NULL,
                subcategory TEXT,
                brand TEXT,
                unit_price REAL,
                cost REAL,
                base_demand INTEGER,
                seasonality_pattern TEXT,
                price_elasticity REAL
            );
            
            CREATE TABLE sim_stores (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                location TEXT NOT NULL,
                region TEXT,
                store_type TEXT,
                square_footage INTEGER,
                customer_traffic_multiplier REAL
            );
            
            CREATE TABLE sim_customers (
                id TEXT PRIMARY KEY,
                segment TEXT NOT NULL,
                location_preference TEXT,
                price_sensitivity REAL,
                brand_loyalty TEXT,
                purchase_frequency REAL
            );
            
            CREATE TABLE sim_sales_transactions (
                transaction_id TEXT PRIMARY KEY,
                product_id TEXT NOT NULL,
                store_id TEXT NOT NULL,
                customer_id TEXT NOT NULL,
                quantity INTEGER,
                unit_price REAL,
                total_amount REAL,
                transaction_date DATETIME,
                promotion_applied BOOLEAN,
                discount_amount REAL,
                FOREIGN KEY (product_id) REFERENCES sim_products(id),
                FOREIGN KEY (store_id) REFERENCES sim_stores(id),
                FOREIGN KEY (customer_id) REFERENCES sim_customers(id)
            );
            
            CREATE TABLE sim_inventory_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                product_id TEXT NOT NULL,
                store_id TEXT NOT NULL,
                current_stock INTEGER,
                reserved_stock INTEGER,
                available_stock INTEGER,
                reorder_point INTEGER,
                max_stock_level INTEGER,
                last_updated DATETIME,
                FOREIGN KEY (product_id) REFERENCES sim_products(id),
                FOREIGN KEY (store_id) REFERENCES sim_stores(id)
            );
            
            CREATE TABLE sim_seasonal_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_name TEXT NOT NULL,
                month INTEGER,
                multiplier REAL,
                description TEXT
            );
            
            CREATE TABLE sim_promotions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                product_id TEXT NOT NULL,
                store_id TEXT,
                start_date DATETIME,
                end_date DATETIME,
                discount_percentage REAL,
                promotion_type TEXT,
                FOREIGN KEY (product_id) REFERENCES sim_products(id)
            );
        """)
        
        conn.commit()
        conn.close()
        logger.info("Simulation database initialized successfully")
    
    def generate_products(self) -> List[Product]:
        """Generate realistic product catalog for simulation"""
        categories = {
            'Electronics': ['Smartphones', 'Laptops', 'Tablets', 'Headphones'],
            'Clothing': ['Shirts', 'Pants', 'Dresses', 'Shoes'],
            'Home & Garden': ['Furniture', 'Appliances', 'Decor', 'Tools'],
            'Food & Beverage': ['Snacks', 'Beverages', 'Frozen', 'Fresh'],
            'Health & Beauty': ['Skincare', 'Makeup', 'Supplements', 'Personal Care']
        }
        
        brands = ['BrandA', 'BrandB', 'BrandC', 'BrandD', 'BrandE']
        seasonality_patterns = ['holiday', 'summer', 'winter', 'back_to_school', 'steady']
        
        products = []
        for i in range(self.config.num_products):
            category = random.choice(list(categories.keys()))
            subcategory = random.choice(categories[category])
            
            product = Product(
                id=f"PROD_{i:04d}",
                name=f"{subcategory} {i}",
                category=category,
                subcategory=subcategory,
                brand=random.choice(brands),
                unit_price=round(random.uniform(10, 500), 2),
                cost=round(random.uniform(5, 250), 2),
                base_demand=random.randint(self.config.base_demand_min, self.config.base_demand_max),
                seasonality_pattern=random.choice(seasonality_patterns),
                price_elasticity=random.uniform(-1.0, -0.1)
            )
            products.append(product)
        
        self.products = products
        self._save_products_to_db()
        logger.info(f"Generated {len(products)} products")
        return products
    
    def generate_stores(self) -> List[Store]:
        """Generate realistic store locations for simulation"""
        regions = ['North', 'South', 'East', 'West', 'Central']
        store_types = ['Flagship', 'Standard', 'Express', 'Outlet']
        
        stores = []
        for i in range(self.config.num_stores):
            region = random.choice(regions)
            store_type = random.choice(store_types)
            
            store = Store(
                id=f"STORE_{i:03d}",
                name=f"{region} {store_type} Store {i}",
                location=f"{region} District",
                region=region,
                store_type=store_type,
                square_footage=random.randint(5000, 50000),
                customer_traffic_multiplier=random.uniform(0.5, 2.0)
            )
            stores.append(store)
        
        self.stores = stores
        self._save_stores_to_db()
        logger.info(f"Generated {len(stores)} stores")
        return stores
    
    def generate_customers(self) -> List[Customer]:
        """Generate realistic customer profiles for simulation"""
        segments = ['Premium', 'Regular', 'Budget', 'Occasional']
        
        customers = []
        for i in range(self.config.num_customers):
            segment = random.choice(segments)
            
            # Generate brand loyalty preferences
            brand_loyalty = {}
            for brand in ['BrandA', 'BrandB', 'BrandC', 'BrandD', 'BrandE']:
                brand_loyalty[brand] = random.uniform(0.1, 0.9)
            
            customer = Customer(
                id=f"CUST_{i:05d}",
                segment=segment,
                location_preference=random.choice(['North', 'South', 'East', 'West', 'Central']),
                price_sensitivity=random.uniform(0.1, 1.0),
                brand_loyalty=brand_loyalty,
                purchase_frequency=random.uniform(0.1, 2.0)
            )
            customers.append(customer)
        
        self.customers = customers
        self._save_customers_to_db()
        logger.info(f"Generated {len(customers)} customers")
        return customers
    
    def simulate_seasonal_demand(self, base_demand: int, pattern: str, day_of_year: int) -> int:
        """Calculate seasonal demand multiplier based on pattern and day of year"""
        seasonal_multipliers = {
            'holiday': self._holiday_pattern(day_of_year),
            'summer': self._summer_pattern(day_of_year),
            'winter': self._winter_pattern(day_of_year),
            'back_to_school': self._back_to_school_pattern(day_of_year),
            'steady': 1.0
        }
        
        multiplier = seasonal_multipliers.get(pattern, 1.0)
        seasonal_demand = int(base_demand * multiplier)
        return max(1, seasonal_demand)
    
    def _holiday_pattern(self, day_of_year: int) -> float:
        """Holiday seasonal pattern (peaks in November-December)"""
        if 320 <= day_of_year <= 365:  # Nov-Dec
            return 1.0 + self.config.seasonal_amplitude * 2
        elif 1 <= day_of_year <= 31:  # January
            return 1.0 + self.config.seasonal_amplitude
        else:
            return 1.0
    
    def _summer_pattern(self, day_of_year: int) -> float:
        """Summer seasonal pattern (peaks in June-August)"""
        if 152 <= day_of_year <= 243:  # Jun-Aug
            return 1.0 + self.config.seasonal_amplitude
        else:
            return 1.0 - self.config.seasonal_amplitude * 0.5
    
    def _winter_pattern(self, day_of_year: int) -> float:
        """Winter seasonal pattern (peaks in December-February)"""
        if day_of_year >= 335 or day_of_year <= 59:  # Dec-Feb
            return 1.0 + self.config.seasonal_amplitude
        else:
            return 1.0 - self.config.seasonal_amplitude * 0.3
    
    def _back_to_school_pattern(self, day_of_year: int) -> float:
        """Back to school pattern (peaks in August-September)"""
        if 213 <= day_of_year <= 273:  # Aug-Sep
            return 1.0 + self.config.seasonal_amplitude
        else:
            return 1.0
    
    def generate_sales_transactions(self, start_date: datetime = None, days: int = None) -> List[SalesTransaction]:
        """Generate realistic sales transactions over specified period"""
        if not start_date:
            start_date = datetime.now() - timedelta(days=self.config.simulation_days)
        if not days:
            days = self.config.simulation_days
        
        transactions = []
        transaction_counter = 0
        
        for day in range(days):
            current_date = start_date + timedelta(days=day)
            day_of_year = current_date.timetuple().tm_yday
            
            # Generate transactions for each store
            for store in self.stores:
                daily_transactions = int(random.uniform(10, 50) * store.customer_traffic_multiplier)
                
                for _ in range(daily_transactions):
                    # Select random product and customer
                    product = random.choice(self.products)
                    customer = random.choice(self.customers)
                    
                    # Calculate seasonal demand
                    seasonal_demand = self.simulate_seasonal_demand(
                        product.base_demand, product.seasonality_pattern, day_of_year
                    )
                    
                    # Determine quantity based on demand and customer behavior
                    base_quantity = max(1, int(seasonal_demand * customer.purchase_frequency / 30))
                    quantity = max(1, int(random.normalvariate(base_quantity, base_quantity * 0.3)))
                    
                    # Apply promotions randomly
                    promotion_applied = random.random() < self.config.promotion_frequency
                    discount_amount = 0.0
                    unit_price = product.unit_price
                    
                    if promotion_applied:
                        discount_percentage = random.uniform(0.1, 0.4)
                        discount_amount = unit_price * discount_percentage
                        unit_price = unit_price * (1 - discount_percentage)
                    
                    # Apply price elasticity effect
                    if promotion_applied:
                        elasticity_effect = abs(product.price_elasticity) * (discount_amount / product.unit_price)
                        quantity = int(quantity * (1 + elasticity_effect))
                    
                    transaction = SalesTransaction(
                        transaction_id=f"TXN_{transaction_counter:08d}",
                        product_id=product.id,
                        store_id=store.id,
                        customer_id=customer.id,
                        quantity=quantity,
                        unit_price=round(unit_price, 2),
                        total_amount=round(unit_price * quantity, 2),
                        transaction_date=current_date + timedelta(
                            hours=random.randint(9, 21),
                            minutes=random.randint(0, 59)
                        ),
                        promotion_applied=promotion_applied,
                        discount_amount=round(discount_amount * quantity, 2)
                    )
                    
                    transactions.append(transaction)
                    transaction_counter += 1
        
        self.sales_transactions = transactions
        self._save_transactions_to_db()
        logger.info(f"Generated {len(transactions)} sales transactions over {days} days")
        return transactions
    
    def generate_inventory_records(self) -> List[InventoryRecord]:
        """Generate realistic inventory records for all product-store combinations"""
        inventory_records = []
        
        for product in self.products:
            for store in self.stores:
                # Calculate inventory levels based on product demand and store size
                base_stock = int(product.base_demand * store.customer_traffic_multiplier * 7)  # 1 week supply
                max_stock = int(base_stock * 3)  # 3 weeks max
                reorder_point = int(base_stock * 0.3)  # Reorder at 30% of base
                
                current_stock = random.randint(reorder_point, max_stock)
                reserved_stock = random.randint(0, int(current_stock * 0.1))
                available_stock = current_stock - reserved_stock
                
                # Simulate stockouts occasionally
                if random.random() < self.config.stockout_probability:
                    current_stock = 0
                    available_stock = 0
                
                inventory_record = InventoryRecord(
                    product_id=product.id,
                    store_id=store.id,
                    current_stock=current_stock,
                    reserved_stock=reserved_stock,
                    available_stock=available_stock,
                    reorder_point=reorder_point,
                    max_stock_level=max_stock,
                    last_updated=datetime.now()
                )
                
                inventory_records.append(inventory_record)
        
        self.inventory_records = inventory_records
        self._save_inventory_to_db()
        logger.info(f"Generated {len(inventory_records)} inventory records")
        return inventory_records
    
    def simulate_market_disruption(self, disruption_type: str, affected_categories: List[str] = None, 
                                 severity: float = 0.5, duration_days: int = 30):
        """Simulate market disruptions like supply chain issues, economic changes, etc."""
        logger.info(f"Simulating {disruption_type} disruption with severity {severity} for {duration_days} days")
        
        disruption_effects = {
            'supply_shortage': self._simulate_supply_shortage,
            'demand_spike': self._simulate_demand_spike,
            'economic_downturn': self._simulate_economic_downturn,
            'seasonal_shift': self._simulate_seasonal_shift,
            'competitor_entry': self._simulate_competitor_entry
        }
        
        if disruption_type in disruption_effects:
            disruption_effects[disruption_type](affected_categories, severity, duration_days)
        else:
            logger.warning(f"Unknown disruption type: {disruption_type}")
    
    def _simulate_supply_shortage(self, categories: List[str], severity: float, duration: int):
        """Simulate supply chain shortage affecting inventory levels"""
        affected_products = [p for p in self.products if not categories or p.category in categories]
        
        for product in affected_products:
            for inventory in self.inventory_records:
                if inventory.product_id == product.id:
                    reduction_factor = severity
                    inventory.current_stock = int(inventory.current_stock * (1 - reduction_factor))
                    inventory.available_stock = max(0, inventory.current_stock - inventory.reserved_stock)
    
    def _simulate_demand_spike(self, categories: List[str], severity: float, duration: int):
        """Simulate sudden demand increase for specific categories"""
        # This would affect future transaction generation
        pass
    
    def _simulate_economic_downturn(self, categories: List[str], severity: float, duration: int):
        """Simulate economic downturn affecting customer purchasing behavior"""
        for customer in self.customers:
            customer.price_sensitivity = min(1.0, customer.price_sensitivity * (1 + severity))
            customer.purchase_frequency = customer.purchase_frequency * (1 - severity * 0.5)
    
    def _simulate_seasonal_shift(self, categories: List[str], severity: float, duration: int):
        """Simulate unexpected seasonal pattern changes"""
        pass
    
    def _simulate_competitor_entry(self, categories: List[str], severity: float, duration: int):
        """Simulate new competitor affecting market share"""
        pass
    
    def get_simulation_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of simulation data"""
        return {
            'config': asdict(self.config),
            'products_count': len(self.products),
            'stores_count': len(self.stores),
            'customers_count': len(self.customers),
            'transactions_count': len(self.sales_transactions),
            'inventory_records_count': len(self.inventory_records),
            'total_revenue': sum(t.total_amount for t in self.sales_transactions),
            'avg_transaction_value': np.mean([t.total_amount for t in self.sales_transactions]) if self.sales_transactions else 0,
            'categories': list(set(p.category for p in self.products)),
            'date_range': {
                'start': min(t.transaction_date for t in self.sales_transactions) if self.sales_transactions else None,
                'end': max(t.transaction_date for t in self.sales_transactions) if self.sales_transactions else None
            }
        }
    
    def export_simulation_data(self, output_dir: str = "simulation_output"):
        """Export simulation data to CSV files for analysis"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Export products
        products_df = pd.DataFrame([asdict(p) for p in self.products])
        products_df.to_csv(output_path / "products.csv", index=False)
        
        # Export stores
        stores_df = pd.DataFrame([asdict(s) for s in self.stores])
        stores_df.to_csv(output_path / "stores.csv", index=False)
        
        # Export customers (without sensitive data)
        customers_data = []
        for c in self.customers:
            customer_dict = asdict(c)
            customer_dict['brand_loyalty'] = json.dumps(customer_dict['brand_loyalty'])
            customers_data.append(customer_dict)
        customers_df = pd.DataFrame(customers_data)
        customers_df.to_csv(output_path / "customers.csv", index=False)
        
        # Export transactions
        transactions_df = pd.DataFrame([asdict(t) for t in self.sales_transactions])
        transactions_df.to_csv(output_path / "sales_transactions.csv", index=False)
        
        # Export inventory
        inventory_df = pd.DataFrame([asdict(i) for i in self.inventory_records])
        inventory_df.to_csv(output_path / "inventory_records.csv", index=False)
        
        # Export summary
        summary = self.get_simulation_summary()
        with open(output_path / "simulation_summary.json", 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Simulation data exported to {output_path}")
    
    def _save_products_to_db(self):
        """Save products to simulation database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for product in self.products:
            cursor.execute("""
                INSERT OR REPLACE INTO sim_products 
                (id, name, category, subcategory, brand, unit_price, cost, base_demand, seasonality_pattern, price_elasticity)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                product.id, product.name, product.category, product.subcategory, product.brand,
                product.unit_price, product.cost, product.base_demand, product.seasonality_pattern, product.price_elasticity
            ))
        
        conn.commit()
        conn.close()
    
    def _save_stores_to_db(self):
        """Save stores to simulation database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for store in self.stores:
            cursor.execute("""
                INSERT OR REPLACE INTO sim_stores 
                (id, name, location, region, store_type, square_footage, customer_traffic_multiplier)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                store.id, store.name, store.location, store.region, store.store_type,
                store.square_footage, store.customer_traffic_multiplier
            ))
        
        conn.commit()
        conn.close()
    
    def _save_customers_to_db(self):
        """Save customers to simulation database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for customer in self.customers:
            cursor.execute("""
                INSERT OR REPLACE INTO sim_customers 
                (id, segment, location_preference, price_sensitivity, brand_loyalty, purchase_frequency)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                customer.id, customer.segment, customer.location_preference, customer.price_sensitivity,
                json.dumps(customer.brand_loyalty), customer.purchase_frequency
            ))
        
        conn.commit()
        conn.close()
    
    def _save_transactions_to_db(self):
        """Save transactions to simulation database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for transaction in self.sales_transactions:
            cursor.execute("""
                INSERT OR REPLACE INTO sim_sales_transactions 
                (transaction_id, product_id, store_id, customer_id, quantity, unit_price, total_amount, 
                 transaction_date, promotion_applied, discount_amount)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                transaction.transaction_id, transaction.product_id, transaction.store_id, transaction.customer_id,
                transaction.quantity, transaction.unit_price, transaction.total_amount, transaction.transaction_date,
                transaction.promotion_applied, transaction.discount_amount
            ))
        
        conn.commit()
        conn.close()
    
    def _save_inventory_to_db(self):
        """Save inventory records to simulation database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for inventory in self.inventory_records:
            cursor.execute("""
                INSERT OR REPLACE INTO sim_inventory_records 
                (product_id, store_id, current_stock, reserved_stock, available_stock, 
                 reorder_point, max_stock_level, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                inventory.product_id, inventory.store_id, inventory.current_stock, inventory.reserved_stock,
                inventory.available_stock, inventory.reorder_point, inventory.max_stock_level, inventory.last_updated
            ))
        
        conn.commit()
        conn.close()
    
    async def run_full_simulation(self) -> Dict[str, Any]:
        """Run complete retail simulation with all components"""
        logger.info("Starting full retail simulation")
        
        # Generate all simulation data
        self.generate_products()
        self.generate_stores()
        self.generate_customers()
        self.generate_sales_transactions()
        self.generate_inventory_records()
        
        # Export data for analysis
        self.export_simulation_data()
        
        summary = self.get_simulation_summary()
        logger.info(f"Simulation completed successfully: {summary}")
        
        return summary
    
    def cleanup(self):
        """Clean up simulation resources"""
        if Path(self.db_path).exists():
            Path(self.db_path).unlink()
        logger.info("Simulation cleanup completed")