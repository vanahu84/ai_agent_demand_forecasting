"""
Sales Data MCP Server for Autonomous Demand Forecasting System.

This server collects and prepares sales transaction data for model training,
including POS and e-commerce data integration, customer behavior analysis,
and data quality validation.
"""

import asyncio
import json
import logging
import os
import sqlite3
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import hashlib

import mcp.server.stdio
from dotenv import load_dotenv

# MCP Server Imports
from mcp import types as mcp_types
from mcp.server.lowlevel import NotificationOptions, Server
from mcp.server.models import InitializationOptions

# Lazy import heavy dependencies
def _get_pandas():
    """Lazy import pandas only when needed."""
    try:
        import pandas as pd
        return pd
    except ImportError:
        logging.warning("pandas not available - some features may be limited")
        return None

def _get_data_models():
    """Lazy import data models only when needed."""
    try:
        from autonomous_demand_forecasting.database.models import (
            SalesTransaction, CustomerBehavior, Product, Store, TrainingData
        )
        return SalesTransaction, CustomerBehavior, Product, Store, TrainingData
    except ImportError:
        logging.warning("Data models not available - using basic functionality")
        return None, None, None, None, None

load_dotenv()

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Console output only
    ]
)

# Log server initialization
logging.info("Initializing Sales Data MCP Server...")

# Database path
DATABASE_PATH = os.path.join(os.path.dirname(__file__), "database", "autonomous_demand_forecasting.db")

# Configuration constants
DEFAULT_DATA_COLLECTION_DAYS = 90
MIN_DATA_QUALITY_SCORE = 0.95
SEASONAL_ANALYSIS_MONTHS = 12


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


def collect_sales_data(days_back: int = DEFAULT_DATA_COLLECTION_DAYS) -> Dict[str, Any]:
    """
    Collect sales transaction data from the past specified days.
    
    Extracts sales transactions from POS and e-commerce systems for model training.
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            cutoff_date = datetime.now() - timedelta(days=days_back)
            
            cursor.execute("""
                SELECT st.*, p.name as product_name, p.brand, s.name as store_name, s.location as region
                FROM sales_transactions st
                JOIN products p ON st.product_id = p.id
                JOIN stores s ON st.store_id = s.id
                WHERE st.transaction_date >= ?
                ORDER BY st.transaction_date DESC
            """, (cutoff_date,))
            
            rows = cursor.fetchall()
        finally:
            return_db_connection(conn)
        
        # Convert to transaction dictionaries (avoid heavy model objects)
        transactions = []
        for row in rows:
            # Parse datetime if it's a string
            transaction_date = row['transaction_date']
            if isinstance(transaction_date, str):
                try:
                    transaction_date = datetime.fromisoformat(transaction_date)
                except ValueError:
                    transaction_date = datetime.now()
            
            transaction = {
                'transaction_id': row['transaction_id'],
                'product_id': row['product_id'],
                'store_id': row['store_id'],
                'category': row['category'],
                'quantity': row['quantity'],
                'unit_price': row['unit_price'],
                'total_amount': row['total_amount'],
                'transaction_date': transaction_date.isoformat(),
                'customer_segment': row['customer_segment'],
                'promotion_applied': bool(row['promotion_applied']),
                'id': row['id']
            }
            transactions.append(transaction)
        
        # Calculate collection statistics
        total_transactions = len(transactions)
        total_revenue = sum(t.total_amount for t in transactions)
        unique_products = len(set(t.product_id for t in transactions))
        unique_categories = len(set(t.category for t in transactions))
        date_range = (
            min(t.transaction_date for t in transactions) if transactions else None,
            max(t.transaction_date for t in transactions) if transactions else None
        )
        
        return {
            "success": True,
            "message": f"Successfully collected {total_transactions} sales transactions",
            "data": {
                "transactions": transactions,
                "statistics": {
                    "total_transactions": total_transactions,
                    "total_revenue": total_revenue,
                    "unique_products": unique_products,
                    "unique_categories": unique_categories,
                    "date_range": {
                        "start": date_range[0].isoformat() if date_range[0] else None,
                        "end": date_range[1].isoformat() if date_range[1] else None
                    },
                    "collection_period_days": days_back
                }
            }
        }
        
    except sqlite3.Error as e:
        logging.error(f"Database error in collect_sales_data: {e}")
        return {
            "success": False,
            "message": f"Database error collecting sales data: {e}"
        }
    except Exception as e:
        logging.error(f"Error in collect_sales_data: {e}")
        return {
            "success": False,
            "message": f"Error collecting sales data: {e}"
        }


def analyze_customer_patterns(segment: Optional[str] = None) -> Dict[str, Any]:
    """
    Analyze customer purchasing patterns and behavior.
    
    Provides insights into customer segments, purchase frequency, and preferences.
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Base query for customer behavior analysis
        base_query = """
            SELECT 
                st.customer_segment,
                st.category,
                COUNT(*) as transaction_count,
                AVG(st.quantity) as avg_quantity,
                AVG(st.unit_price) as avg_unit_price,
                AVG(st.total_amount) as avg_transaction_value,
                SUM(st.total_amount) as total_spent,
                COUNT(DISTINCT st.product_id) as unique_products,
                AVG(CASE WHEN st.promotion_applied THEN 1 ELSE 0 END) as promotion_usage_rate
            FROM sales_transactions st
            JOIN products p ON st.product_id = p.id
            WHERE st.transaction_date >= ?
        """
        
        params = [datetime.now() - timedelta(days=DEFAULT_DATA_COLLECTION_DAYS)]
        
        if segment:
            base_query += " AND st.customer_segment = ?"
            params.append(segment)
        
        base_query += """
            GROUP BY st.customer_segment, st.category
            ORDER BY st.customer_segment, total_spent DESC
        """
        
        cursor.execute(base_query, params)
        rows = cursor.fetchall()
        
        # Process results into customer behavior patterns
        customer_patterns = {}
        for row in rows:
            segment_key = row['customer_segment'] or 'Unknown'
            if segment_key not in customer_patterns:
                customer_patterns[segment_key] = {
                    'categories': {},
                    'total_transactions': 0,
                    'total_spent': 0,
                    'avg_transaction_value': 0
                }
            
            category_data = {
                'transaction_count': row['transaction_count'],
                'avg_quantity': row['avg_quantity'],
                'avg_unit_price': row['avg_unit_price'],
                'avg_transaction_value': row['avg_transaction_value'],
                'total_spent': row['total_spent'],
                'unique_products': row['unique_products'],
                'promotion_usage_rate': row['promotion_usage_rate']
            }
            
            customer_patterns[segment_key]['categories'][row['category']] = category_data
            customer_patterns[segment_key]['total_transactions'] += row['transaction_count']
            customer_patterns[segment_key]['total_spent'] += row['total_spent']
        
        # Calculate segment-level averages
        for segment_key, data in customer_patterns.items():
            if data['total_transactions'] > 0:
                data['avg_transaction_value'] = data['total_spent'] / data['total_transactions']
        
        # Get purchase frequency analysis
        frequency_query = """
            SELECT 
                customer_segment,
                COUNT(DISTINCT DATE(transaction_date)) as active_days,
                COUNT(*) as total_transactions,
                MIN(transaction_date) as first_purchase,
                MAX(transaction_date) as last_purchase
            FROM sales_transactions
            WHERE transaction_date >= ?
        """
        
        frequency_params = [datetime.now() - timedelta(days=DEFAULT_DATA_COLLECTION_DAYS)]
        
        if segment:
            frequency_query += " AND customer_segment = ?"
            frequency_params.append(segment)
        
        frequency_query += " GROUP BY customer_segment"
        
        cursor.execute(frequency_query, frequency_params)
        frequency_rows = cursor.fetchall()
        
        # Add frequency data to patterns
        for row in frequency_rows:
            segment_key = row['customer_segment'] or 'Unknown'
            if segment_key in customer_patterns:
                days_active = row['active_days']
                total_transactions = row['total_transactions']
                
                customer_patterns[segment_key]['frequency_analysis'] = {
                    'active_days': days_active,
                    'avg_transactions_per_day': total_transactions / days_active if days_active > 0 else 0,
                    'purchase_frequency': days_active / DEFAULT_DATA_COLLECTION_DAYS,
                    'first_purchase': row['first_purchase'],
                    'last_purchase': row['last_purchase']
                }
        
        conn.close()
        
        return {
            "success": True,
            "message": f"Successfully analyzed customer patterns for {len(customer_patterns)} segments",
            "data": {
                "customer_patterns": customer_patterns,
                "analysis_period_days": DEFAULT_DATA_COLLECTION_DAYS,
                "analyzed_at": datetime.now().isoformat()
            }
        }
        
    except sqlite3.Error as e:
        logging.error(f"Database error in analyze_customer_patterns: {e}")
        return {
            "success": False,
            "message": f"Database error analyzing customer patterns: {e}"
        }
    except Exception as e:
        logging.error(f"Error in analyze_customer_patterns: {e}")
        return {
            "success": False,
            "message": f"Error analyzing customer patterns: {e}"
        }


def extract_seasonal_trends(category: Optional[str] = None) -> Dict[str, Any]:
    """
    Extract seasonal trends and patterns from sales data.
    
    Analyzes seasonal variations, monthly patterns, and cyclical behavior.
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Query for seasonal analysis (monthly aggregation)
        seasonal_query = """
            SELECT 
                category,
                strftime('%Y', transaction_date) as year,
                strftime('%m', transaction_date) as month,
                COUNT(*) as transaction_count,
                SUM(quantity) as total_quantity,
                SUM(total_amount) as total_revenue,
                AVG(total_amount) as avg_transaction_value,
                COUNT(DISTINCT product_id) as unique_products
            FROM sales_transactions
            WHERE transaction_date >= ?
        """
        
        params = [datetime.now() - timedelta(days=SEASONAL_ANALYSIS_MONTHS * 30)]
        
        if category:
            seasonal_query += " AND category = ?"
            params.append(category)
        
        seasonal_query += """
            GROUP BY category, year, month
            ORDER BY category, year, month
        """
        
        cursor.execute(seasonal_query, params)
        rows = cursor.fetchall()
        
        # Process seasonal data
        seasonal_trends = {}
        for row in rows:
            cat = row['category']
            if cat not in seasonal_trends:
                seasonal_trends[cat] = {
                    'monthly_data': [],
                    'yearly_totals': {},
                    'seasonal_patterns': {}
                }
            
            month_data = {
                'year': int(row['year']),
                'month': int(row['month']),
                'transaction_count': row['transaction_count'],
                'total_quantity': row['total_quantity'],
                'total_revenue': row['total_revenue'],
                'avg_transaction_value': row['avg_transaction_value'],
                'unique_products': row['unique_products']
            }
            
            seasonal_trends[cat]['monthly_data'].append(month_data)
            
            # Aggregate yearly totals
            year = int(row['year'])
            if year not in seasonal_trends[cat]['yearly_totals']:
                seasonal_trends[cat]['yearly_totals'][year] = {
                    'total_revenue': 0,
                    'total_transactions': 0,
                    'total_quantity': 0
                }
            
            seasonal_trends[cat]['yearly_totals'][year]['total_revenue'] += row['total_revenue']
            seasonal_trends[cat]['yearly_totals'][year]['total_transactions'] += row['transaction_count']
            seasonal_trends[cat]['yearly_totals'][year]['total_quantity'] += row['total_quantity']
        
        # Calculate seasonal patterns and multipliers
        for cat, data in seasonal_trends.items():
            monthly_revenues = {}
            for month_data in data['monthly_data']:
                month = month_data['month']
                if month not in monthly_revenues:
                    monthly_revenues[month] = []
                monthly_revenues[month].append(month_data['total_revenue'])
            
            # Calculate average monthly revenue and seasonal multipliers
            avg_monthly_revenues = {}
            seasonal_multipliers = {}
            
            for month, revenues in monthly_revenues.items():
                avg_monthly_revenues[month] = statistics.mean(revenues)
            
            if avg_monthly_revenues:
                overall_avg = statistics.mean(avg_monthly_revenues.values())
                for month, avg_revenue in avg_monthly_revenues.items():
                    seasonal_multipliers[month] = avg_revenue / overall_avg if overall_avg > 0 else 1.0
            
            data['seasonal_patterns'] = {
                'avg_monthly_revenues': avg_monthly_revenues,
                'seasonal_multipliers': seasonal_multipliers,
                'peak_month': max(seasonal_multipliers.items(), key=lambda x: x[1])[0] if seasonal_multipliers else None,
                'low_month': min(seasonal_multipliers.items(), key=lambda x: x[1])[0] if seasonal_multipliers else None
            }
        
        conn.close()
        
        return {
            "success": True,
            "message": f"Successfully extracted seasonal trends for {len(seasonal_trends)} categories",
            "data": {
                "seasonal_trends": seasonal_trends,
                "analysis_period_months": SEASONAL_ANALYSIS_MONTHS,
                "analyzed_at": datetime.now().isoformat()
            }
        }
        
    except sqlite3.Error as e:
        logging.error(f"Database error in extract_seasonal_trends: {e}")
        return {
            "success": False,
            "message": f"Database error extracting seasonal trends: {e}"
        }
    except Exception as e:
        logging.error(f"Error in extract_seasonal_trends: {e}")
        return {
            "success": False,
            "message": f"Error extracting seasonal trends: {e}"
        }


def clean_and_transform_data(dataset_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Implement automated data cleaning and transformation functions.
    
    Performs data cleaning operations including outlier removal, missing value imputation,
    and data standardization.
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get recent data for cleaning
        cutoff_date = datetime.now() - timedelta(days=DEFAULT_DATA_COLLECTION_DAYS)
        
        # First, identify records that need cleaning
        cursor.execute("""
            SELECT 
                id,
                transaction_id,
                product_id,
                store_id,
                category,
                quantity,
                unit_price,
                total_amount,
                transaction_date,
                customer_segment,
                promotion_applied
            FROM sales_transactions
            WHERE transaction_date >= ?
            ORDER BY transaction_date DESC
        """, (cutoff_date,))
        
        rows = cursor.fetchall()
        
        if not rows:
            return {
                "success": True,
                "message": "No data available for cleaning",
                "data": {
                    "records_processed": 0,
                    "records_cleaned": 0,
                    "cleaning_operations": []
                }
            }
        
        cleaning_operations = []
        records_cleaned = 0
        
        # Process each record for cleaning
        for row in rows:
            record_id = row['id']
            needs_update = False
            updates = {}
            operations_for_record = []
            
            # 1. Fix calculation errors (total_amount should equal quantity * unit_price)
            expected_total = row['quantity'] * row['unit_price']
            if abs(row['total_amount'] - expected_total) > 0.01:
                updates['total_amount'] = expected_total
                operations_for_record.append(f"Fixed calculation error: {row['total_amount']} -> {expected_total}")
                needs_update = True
            
            # 2. Handle missing customer segments
            if not row['customer_segment'] or row['customer_segment'].strip() == '':
                # Assign segment based on transaction value
                if row['total_amount'] > 500:
                    updates['customer_segment'] = 'Premium'
                elif row['total_amount'] > 100:
                    updates['customer_segment'] = 'Standard'
                else:
                    updates['customer_segment'] = 'Budget'
                operations_for_record.append(f"Assigned customer segment: {updates['customer_segment']}")
                needs_update = True
            
            # 3. Standardize category names (capitalize first letter)
            if row['category']:
                standardized_category = row['category'].strip().title()
                if standardized_category != row['category']:
                    updates['category'] = standardized_category
                    operations_for_record.append(f"Standardized category: {row['category']} -> {standardized_category}")
                    needs_update = True
            
            # 4. Handle outliers in quantity (cap at reasonable maximum)
            max_reasonable_quantity = 100
            if row['quantity'] > max_reasonable_quantity:
                updates['quantity'] = max_reasonable_quantity
                updates['total_amount'] = max_reasonable_quantity * row['unit_price']
                operations_for_record.append(f"Capped quantity outlier: {row['quantity']} -> {max_reasonable_quantity}")
                needs_update = True
            
            # 5. Handle outliers in unit price (cap at reasonable maximum)
            max_reasonable_price = 10000
            if row['unit_price'] > max_reasonable_price:
                updates['unit_price'] = max_reasonable_price
                updates['total_amount'] = row['quantity'] * max_reasonable_price
                operations_for_record.append(f"Capped price outlier: {row['unit_price']} -> {max_reasonable_price}")
                needs_update = True
            
            # Apply updates if needed
            if needs_update:
                # Build update query dynamically
                set_clauses = []
                update_values = []
                
                for field, value in updates.items():
                    set_clauses.append(f"{field} = ?")
                    update_values.append(value)
                
                update_values.append(record_id)
                
                update_query = f"""
                    UPDATE sales_transactions 
                    SET {', '.join(set_clauses)}
                    WHERE id = ?
                """
                
                cursor.execute(update_query, update_values)
                records_cleaned += 1
                
                cleaning_operations.append({
                    "record_id": record_id,
                    "transaction_id": row['transaction_id'],
                    "operations": operations_for_record
                })
        
        # Remove duplicate transactions (keep the first occurrence)
        cursor.execute("""
            DELETE FROM sales_transactions 
            WHERE id NOT IN (
                SELECT MIN(id) 
                FROM sales_transactions 
                WHERE transaction_date >= ?
                GROUP BY transaction_id
            ) AND transaction_date >= ?
        """, (cutoff_date, cutoff_date))
        
        duplicates_removed = cursor.rowcount
        if duplicates_removed > 0:
            cleaning_operations.append({
                "operation": "duplicate_removal",
                "records_removed": duplicates_removed
            })
        
        conn.commit()
        conn.close()
        
        return {
            "success": True,
            "message": f"Data cleaning completed. {records_cleaned} records cleaned, {duplicates_removed} duplicates removed",
            "data": {
                "records_processed": len(rows),
                "records_cleaned": records_cleaned,
                "duplicates_removed": duplicates_removed,
                "cleaning_operations": cleaning_operations,
                "cleaned_at": datetime.now().isoformat()
            }
        }
        
    except sqlite3.Error as e:
        logging.error(f"Database error in clean_and_transform_data: {e}")
        return {
            "success": False,
            "message": f"Database error cleaning data: {e}"
        }
    except Exception as e:
        logging.error(f"Error in clean_and_transform_data: {e}")
        return {
            "success": False,
            "message": f"Error cleaning data: {e}"
        }


def assess_data_completeness(days_back: int = DEFAULT_DATA_COLLECTION_DAYS) -> Dict[str, Any]:
    """
    Assess data completeness with detailed completeness checks.
    
    Provides comprehensive analysis of data completeness across different dimensions.
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        # Overall completeness assessment
        cursor.execute("""
            SELECT 
                COUNT(*) as total_records,
                COUNT(CASE WHEN transaction_id IS NOT NULL AND transaction_id != '' THEN 1 END) as complete_transaction_id,
                COUNT(CASE WHEN product_id IS NOT NULL AND product_id != '' THEN 1 END) as complete_product_id,
                COUNT(CASE WHEN store_id IS NOT NULL AND store_id != '' THEN 1 END) as complete_store_id,
                COUNT(CASE WHEN category IS NOT NULL AND category != '' THEN 1 END) as complete_category,
                COUNT(CASE WHEN quantity IS NOT NULL AND quantity > 0 THEN 1 END) as complete_quantity,
                COUNT(CASE WHEN unit_price IS NOT NULL AND unit_price > 0 THEN 1 END) as complete_unit_price,
                COUNT(CASE WHEN total_amount IS NOT NULL AND total_amount > 0 THEN 1 END) as complete_total_amount,
                COUNT(CASE WHEN transaction_date IS NOT NULL THEN 1 END) as complete_transaction_date,
                COUNT(CASE WHEN customer_segment IS NOT NULL AND customer_segment != '' THEN 1 END) as complete_customer_segment
            FROM sales_transactions
            WHERE transaction_date >= ?
        """, (cutoff_date,))
        
        completeness_row = cursor.fetchone()
        total_records = completeness_row['total_records']
        
        if total_records == 0:
            return {
                "success": True,
                "message": "No data available for completeness assessment",
                "data": {
                    "total_records": 0,
                    "completeness_scores": {},
                    "recommendations": ["No data available for analysis"]
                }
            }
        
        # Calculate completeness scores for each field
        completeness_scores = {}
        required_fields = [
            'transaction_id', 'product_id', 'store_id', 'category', 
            'quantity', 'unit_price', 'total_amount', 'transaction_date'
        ]
        optional_fields = ['customer_segment']
        
        for field in required_fields:
            complete_count = completeness_row[f'complete_{field}']
            completeness_scores[field] = {
                'completeness_rate': complete_count / total_records,
                'complete_records': complete_count,
                'missing_records': total_records - complete_count,
                'required': True
            }
        
        for field in optional_fields:
            complete_count = completeness_row[f'complete_{field}']
            completeness_scores[field] = {
                'completeness_rate': complete_count / total_records,
                'complete_records': complete_count,
                'missing_records': total_records - complete_count,
                'required': False
            }
        
        # Completeness by category
        cursor.execute("""
            SELECT 
                category,
                COUNT(*) as category_total,
                COUNT(CASE WHEN customer_segment IS NOT NULL AND customer_segment != '' THEN 1 END) as complete_segments
            FROM sales_transactions
            WHERE transaction_date >= ?
            GROUP BY category
            ORDER BY category_total DESC
        """, (cutoff_date,))
        
        category_completeness = {}
        for row in cursor.fetchall():
            category = row['category']
            category_completeness[category] = {
                'total_records': row['category_total'],
                'customer_segment_completeness': row['complete_segments'] / row['category_total']
            }
        
        # Completeness over time (daily)
        cursor.execute("""
            SELECT 
                DATE(transaction_date) as date,
                COUNT(*) as daily_total,
                COUNT(CASE WHEN customer_segment IS NOT NULL AND customer_segment != '' THEN 1 END) as complete_segments
            FROM sales_transactions
            WHERE transaction_date >= ?
            GROUP BY DATE(transaction_date)
            ORDER BY date DESC
            LIMIT 30
        """, (cutoff_date,))
        
        daily_completeness = {}
        for row in cursor.fetchall():
            date = row['date']
            daily_completeness[date] = {
                'total_records': row['daily_total'],
                'customer_segment_completeness': row['complete_segments'] / row['daily_total'] if row['daily_total'] > 0 else 0
            }
        
        conn.close()
        
        # Generate recommendations
        recommendations = []
        
        # Check for critical missing data
        for field, scores in completeness_scores.items():
            if scores['required'] and scores['completeness_rate'] < 0.95:
                recommendations.append(f"Critical: {field} completeness is {scores['completeness_rate']:.1%} - investigate data source")
            elif scores['completeness_rate'] < 0.80:
                recommendations.append(f"Warning: {field} completeness is {scores['completeness_rate']:.1%} - consider data enrichment")
        
        # Check for category-specific issues
        for category, scores in category_completeness.items():
            if scores['customer_segment_completeness'] < 0.50:
                recommendations.append(f"Customer segment data missing for {category} category ({scores['customer_segment_completeness']:.1%} complete)")
        
        # Overall assessment
        required_completeness = [scores['completeness_rate'] for field, scores in completeness_scores.items() if scores['required']]
        overall_completeness = statistics.mean(required_completeness) if required_completeness else 0
        
        if overall_completeness > 0.95:
            recommendations.append("Excellent data completeness - no immediate action required")
        elif overall_completeness > 0.90:
            recommendations.append("Good data completeness - monitor for trends")
        elif overall_completeness > 0.80:
            recommendations.append("Fair data completeness - implement data quality improvements")
        else:
            recommendations.append("Poor data completeness - urgent data quality initiative required")
        
        return {
            "success": True,
            "message": f"Completeness assessment completed for {total_records} records",
            "data": {
                "total_records": total_records,
                "overall_completeness": overall_completeness,
                "completeness_scores": completeness_scores,
                "category_completeness": category_completeness,
                "daily_completeness": daily_completeness,
                "recommendations": recommendations,
                "assessed_at": datetime.now().isoformat()
            }
        }
        
    except sqlite3.Error as e:
        logging.error(f"Database error in assess_data_completeness: {e}")
        return {
            "success": False,
            "message": f"Database error assessing completeness: {e}"
        }
    except Exception as e:
        logging.error(f"Error in assess_data_completeness: {e}")
        return {
            "success": False,
            "message": f"Error assessing completeness: {e}"
        }


def validate_data_quality(dataset_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate data quality with completeness and consistency checks.
    
    Performs comprehensive data quality assessment on collected sales data.
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get recent data for quality assessment
        cutoff_date = datetime.now() - timedelta(days=DEFAULT_DATA_COLLECTION_DAYS)
        
        # Check data completeness
        cursor.execute("""
            SELECT 
                COUNT(*) as total_records,
                COUNT(CASE WHEN transaction_id IS NULL OR transaction_id = '' THEN 1 END) as missing_transaction_id,
                COUNT(CASE WHEN product_id IS NULL OR product_id = '' THEN 1 END) as missing_product_id,
                COUNT(CASE WHEN store_id IS NULL OR store_id = '' THEN 1 END) as missing_store_id,
                COUNT(CASE WHEN category IS NULL OR category = '' THEN 1 END) as missing_category,
                COUNT(CASE WHEN quantity IS NULL OR quantity <= 0 THEN 1 END) as invalid_quantity,
                COUNT(CASE WHEN unit_price IS NULL OR unit_price <= 0 THEN 1 END) as invalid_unit_price,
                COUNT(CASE WHEN total_amount IS NULL OR total_amount <= 0 THEN 1 END) as invalid_total_amount,
                COUNT(CASE WHEN transaction_date IS NULL THEN 1 END) as missing_transaction_date
            FROM sales_transactions
            WHERE transaction_date >= ?
        """, (cutoff_date,))
        
        completeness_row = cursor.fetchone()
        
        # Check data consistency
        cursor.execute("""
            SELECT 
                COUNT(*) as total_records,
                COUNT(CASE WHEN ABS(total_amount - (quantity * unit_price)) > 0.01 THEN 1 END) as amount_calculation_errors,
                COUNT(CASE WHEN quantity > 1000 THEN 1 END) as suspicious_quantities,
                COUNT(CASE WHEN unit_price > 10000 THEN 1 END) as suspicious_prices,
                COUNT(CASE WHEN total_amount > 100000 THEN 1 END) as suspicious_amounts
            FROM sales_transactions
            WHERE transaction_date >= ?
        """, (cutoff_date,))
        
        consistency_row = cursor.fetchone()
        
        # Check for duplicate transactions
        cursor.execute("""
            SELECT COUNT(*) as duplicate_count
            FROM (
                SELECT transaction_id, COUNT(*) as cnt
                FROM sales_transactions
                WHERE transaction_date >= ?
                GROUP BY transaction_id
                HAVING COUNT(*) > 1
            )
        """, (cutoff_date,))
        
        duplicate_count = cursor.fetchone()['duplicate_count']
        
        # Check referential integrity
        cursor.execute("""
            SELECT 
                COUNT(CASE WHEN p.id IS NULL THEN 1 END) as orphaned_products,
                COUNT(CASE WHEN s.id IS NULL THEN 1 END) as orphaned_stores
            FROM sales_transactions st
            LEFT JOIN products p ON st.product_id = p.id
            LEFT JOIN stores s ON st.store_id = s.id
            WHERE st.transaction_date >= ?
        """, (cutoff_date,))
        
        integrity_row = cursor.fetchone()
        
        conn.close()
        
        # Calculate quality scores
        total_records = completeness_row['total_records']
        
        if total_records == 0:
            return {
                "success": True,
                "message": "No data available for quality assessment",
                "data": {
                    "quality_score": 0.0,
                    "total_records": 0,
                    "issues": ["No data available"]
                }
            }
        
        # Completeness score (percentage of complete records)
        completeness_issues = (
            completeness_row['missing_transaction_id'] +
            completeness_row['missing_product_id'] +
            completeness_row['missing_store_id'] +
            completeness_row['missing_category'] +
            completeness_row['invalid_quantity'] +
            completeness_row['invalid_unit_price'] +
            completeness_row['invalid_total_amount'] +
            completeness_row['missing_transaction_date']
        )
        
        completeness_score = max(0, (total_records - completeness_issues) / total_records)
        
        # Consistency score
        consistency_issues = (
            consistency_row['amount_calculation_errors'] +
            consistency_row['suspicious_quantities'] +
            consistency_row['suspicious_prices'] +
            consistency_row['suspicious_amounts']
        )
        
        consistency_score = max(0, (total_records - consistency_issues) / total_records)
        
        # Integrity score
        integrity_issues = (
            integrity_row['orphaned_products'] +
            integrity_row['orphaned_stores'] +
            duplicate_count
        )
        
        integrity_score = max(0, (total_records - integrity_issues) / total_records)
        
        # Overall quality score (weighted average)
        overall_quality_score = (
            completeness_score * 0.4 +
            consistency_score * 0.4 +
            integrity_score * 0.2
        )
        
        # Identify specific issues
        issues = []
        if completeness_row['missing_transaction_id'] > 0:
            issues.append(f"{completeness_row['missing_transaction_id']} records missing transaction ID")
        if completeness_row['missing_product_id'] > 0:
            issues.append(f"{completeness_row['missing_product_id']} records missing product ID")
        if completeness_row['missing_store_id'] > 0:
            issues.append(f"{completeness_row['missing_store_id']} records missing store ID")
        if completeness_row['invalid_quantity'] > 0:
            issues.append(f"{completeness_row['invalid_quantity']} records with invalid quantity")
        if consistency_row['amount_calculation_errors'] > 0:
            issues.append(f"{consistency_row['amount_calculation_errors']} records with calculation errors")
        if duplicate_count > 0:
            issues.append(f"{duplicate_count} duplicate transaction IDs found")
        if integrity_row['orphaned_products'] > 0:
            issues.append(f"{integrity_row['orphaned_products']} records reference non-existent products")
        if integrity_row['orphaned_stores'] > 0:
            issues.append(f"{integrity_row['orphaned_stores']} records reference non-existent stores")
        
        # Quality assessment
        quality_level = "EXCELLENT" if overall_quality_score >= 0.95 else \
                       "GOOD" if overall_quality_score >= 0.90 else \
                       "FAIR" if overall_quality_score >= 0.80 else \
                       "POOR"
        
        return {
            "success": True,
            "message": f"Data quality assessment completed - {quality_level} quality",
            "data": {
                "quality_score": overall_quality_score,
                "quality_level": quality_level,
                "total_records": total_records,
                "scores": {
                    "completeness": completeness_score,
                    "consistency": consistency_score,
                    "integrity": integrity_score
                },
                "issues": issues,
                "detailed_metrics": {
                    "completeness": dict(completeness_row),
                    "consistency": dict(consistency_row),
                    "integrity": dict(integrity_row),
                    "duplicates": duplicate_count
                },
                "assessed_at": datetime.now().isoformat()
            }
        }
        
    except sqlite3.Error as e:
        logging.error(f"Database error in validate_data_quality: {e}")
        return {
            "success": False,
            "message": f"Database error validating data quality: {e}"
        }
    except Exception as e:
        logging.error(f"Error in validate_data_quality: {e}")
        return {
            "success": False,
            "message": f"Error validating data quality: {e}"
        }


def segment_customers_by_behavior(days_back: int = DEFAULT_DATA_COLLECTION_DAYS) -> Dict[str, Any]:
    """
    Implement customer segmentation algorithms based on purchase patterns.
    
    Segments customers using RFM analysis (Recency, Frequency, Monetary) and behavioral patterns.
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        # Calculate RFM metrics for customer segmentation
        cursor.execute("""
            SELECT 
                COALESCE(customer_segment, 'Unknown') as original_segment,
                COUNT(*) as frequency,
                SUM(total_amount) as monetary_value,
                AVG(total_amount) as avg_transaction_value,
                MAX(transaction_date) as last_purchase_date,
                MIN(transaction_date) as first_purchase_date,
                COUNT(DISTINCT category) as category_diversity,
                AVG(CASE WHEN promotion_applied THEN 1 ELSE 0 END) as promotion_usage_rate,
                COUNT(DISTINCT DATE(transaction_date)) as active_days
            FROM sales_transactions
            WHERE transaction_date >= ?
            GROUP BY COALESCE(customer_segment, 'Unknown')
            HAVING COUNT(*) > 0
        """, (cutoff_date,))
        
        segment_data = cursor.fetchall()
        
        if not segment_data:
            return {
                "success": True,
                "message": "No customer data available for segmentation",
                "data": {
                    "segments": {},
                    "segmentation_criteria": {},
                    "recommendations": ["No customer data available for analysis"]
                }
            }
        
        # Calculate segmentation thresholds based on data distribution
        monetary_values = [row['monetary_value'] for row in segment_data]
        frequency_values = [row['frequency'] for row in segment_data]
        
        # Calculate percentiles for segmentation
        monetary_values.sort()
        frequency_values.sort()
        
        monetary_high = monetary_values[int(len(monetary_values) * 0.8)] if len(monetary_values) > 5 else max(monetary_values)
        monetary_low = monetary_values[int(len(monetary_values) * 0.3)] if len(monetary_values) > 5 else min(monetary_values)
        
        frequency_high = frequency_values[int(len(frequency_values) * 0.8)] if len(frequency_values) > 5 else max(frequency_values)
        frequency_low = frequency_values[int(len(frequency_values) * 0.3)] if len(frequency_values) > 5 else min(frequency_values)
        
        # Segment customers based on behavioral patterns
        behavioral_segments = {}
        
        for row in segment_data:
            original_segment = row['original_segment']
            monetary = row['monetary_value']
            frequency = row['frequency']
            recency_days = (datetime.now() - datetime.fromisoformat(row['last_purchase_date'])).days
            
            # Determine behavioral segment
            if monetary >= monetary_high and frequency >= frequency_high:
                behavioral_segment = "Champions"
            elif monetary >= monetary_high and frequency < frequency_high:
                behavioral_segment = "Potential Loyalists"
            elif monetary < monetary_low and frequency >= frequency_high:
                behavioral_segment = "Loyal Customers"
            elif monetary >= monetary_high and recency_days > 30:
                behavioral_segment = "At Risk"
            elif monetary < monetary_low and frequency < frequency_low:
                behavioral_segment = "New Customers"
            elif recency_days > 60:
                behavioral_segment = "Cannot Lose Them"
            else:
                behavioral_segment = "Regular Customers"
            
            # Calculate customer lifetime value (CLV) estimate
            avg_days_between_purchases = days_back / row['active_days'] if row['active_days'] > 0 else days_back
            estimated_annual_frequency = 365 / avg_days_between_purchases if avg_days_between_purchases > 0 else 1
            estimated_clv = row['avg_transaction_value'] * estimated_annual_frequency * 2  # 2-year estimate
            
            behavioral_segments[original_segment] = {
                "original_segment": original_segment,
                "behavioral_segment": behavioral_segment,
                "metrics": {
                    "frequency": row['frequency'],
                    "monetary_value": row['monetary_value'],
                    "avg_transaction_value": row['avg_transaction_value'],
                    "recency_days": recency_days,
                    "category_diversity": row['category_diversity'],
                    "promotion_usage_rate": row['promotion_usage_rate'],
                    "active_days": row['active_days'],
                    "estimated_clv": estimated_clv
                },
                "characteristics": _get_segment_characteristics(behavioral_segment),
                "recommendations": _get_segment_recommendations(behavioral_segment)
            }
        
        # Calculate category preferences for each segment
        cursor.execute("""
            SELECT 
                COALESCE(customer_segment, 'Unknown') as segment,
                category,
                COUNT(*) as purchase_count,
                SUM(total_amount) as category_spend,
                AVG(total_amount) as avg_category_spend
            FROM sales_transactions
            WHERE transaction_date >= ?
            GROUP BY COALESCE(customer_segment, 'Unknown'), category
            ORDER BY segment, category_spend DESC
        """, (cutoff_date,))
        
        category_preferences = {}
        for row in cursor.fetchall():
            segment = row['segment']
            if segment not in category_preferences:
                category_preferences[segment] = []
            
            category_preferences[segment].append({
                "category": row['category'],
                "purchase_count": row['purchase_count'],
                "total_spend": row['category_spend'],
                "avg_spend": row['avg_category_spend']
            })
        
        # Add category preferences to segments
        for segment_name, segment_info in behavioral_segments.items():
            segment_info["category_preferences"] = category_preferences.get(segment_name, [])
        
        conn.close()
        
        # Generate overall segmentation insights
        segmentation_criteria = {
            "monetary_thresholds": {
                "high": monetary_high,
                "low": monetary_low
            },
            "frequency_thresholds": {
                "high": frequency_high,
                "low": frequency_low
            },
            "recency_thresholds": {
                "recent": 30,
                "at_risk": 60
            }
        }
        
        # Generate recommendations
        recommendations = []
        champion_count = sum(1 for s in behavioral_segments.values() if s["behavioral_segment"] == "Champions")
        at_risk_count = sum(1 for s in behavioral_segments.values() if s["behavioral_segment"] == "At Risk")
        
        if champion_count > 0:
            recommendations.append(f"Focus on retaining {champion_count} Champion customers with exclusive offers")
        if at_risk_count > 0:
            recommendations.append(f"Implement win-back campaigns for {at_risk_count} At Risk customers")
        
        recommendations.append("Use behavioral segments for targeted marketing campaigns")
        recommendations.append("Monitor segment migration patterns for early intervention")
        
        return {
            "success": True,
            "message": f"Customer segmentation completed for {len(behavioral_segments)} segments",
            "data": {
                "segments": behavioral_segments,
                "segmentation_criteria": segmentation_criteria,
                "recommendations": recommendations,
                "analysis_period_days": days_back,
                "segmented_at": datetime.now().isoformat()
            }
        }
        
    except sqlite3.Error as e:
        logging.error(f"Database error in segment_customers_by_behavior: {e}")
        return {
            "success": False,
            "message": f"Database error segmenting customers: {e}"
        }
    except Exception as e:
        logging.error(f"Error in segment_customers_by_behavior: {e}")
        return {
            "success": False,
            "message": f"Error segmenting customers: {e}"
        }


def _get_segment_characteristics(segment: str) -> List[str]:
    """Get characteristics for a behavioral segment."""
    characteristics = {
        "Champions": ["High value", "High frequency", "Recent purchases", "Brand advocates"],
        "Potential Loyalists": ["High value", "Low frequency", "Good potential", "Need engagement"],
        "Loyal Customers": ["High frequency", "Lower value", "Consistent buyers", "Price sensitive"],
        "At Risk": ["High value", "Declining activity", "Need attention", "Churn risk"],
        "New Customers": ["Low frequency", "Recent acquisition", "Growth potential", "Need nurturing"],
        "Cannot Lose Them": ["High value", "Inactive", "Critical retention", "Win-back priority"],
        "Regular Customers": ["Moderate value", "Moderate frequency", "Stable segment", "Upsell potential"]
    }
    return characteristics.get(segment, ["Standard customer profile"])


def _get_segment_recommendations(segment: str) -> List[str]:
    """Get marketing recommendations for a behavioral segment."""
    recommendations = {
        "Champions": ["Exclusive VIP programs", "Early access to new products", "Referral incentives"],
        "Potential Loyalists": ["Loyalty program enrollment", "Personalized offers", "Engagement campaigns"],
        "Loyal Customers": ["Volume discounts", "Category-specific promotions", "Appreciation rewards"],
        "At Risk": ["Win-back campaigns", "Special discounts", "Personal outreach"],
        "New Customers": ["Welcome series", "Product education", "First-purchase incentives"],
        "Cannot Lose Them": ["Urgent retention campaigns", "High-value offers", "Personal contact"],
        "Regular Customers": ["Cross-sell campaigns", "Category expansion", "Seasonal promotions"]
    }
    return recommendations.get(segment, ["Standard marketing approach"])


def analyze_promotional_impact(days_back: int = DEFAULT_DATA_COLLECTION_DAYS) -> Dict[str, Any]:
    """
    Add promotional impact analysis and seasonal adjustment tools.
    
    Analyzes the effectiveness of promotions and their impact on sales patterns.
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        # Overall promotional impact analysis
        cursor.execute("""
            SELECT 
                promotion_applied,
                COUNT(*) as transaction_count,
                SUM(total_amount) as total_revenue,
                AVG(total_amount) as avg_transaction_value,
                AVG(quantity) as avg_quantity,
                COUNT(DISTINCT product_id) as unique_products,
                COUNT(DISTINCT customer_segment) as customer_segments
            FROM sales_transactions
            WHERE transaction_date >= ?
            GROUP BY promotion_applied
        """, (cutoff_date,))
        
        promo_overview = {}
        for row in cursor.fetchall():
            promo_type = "With Promotion" if row['promotion_applied'] else "Without Promotion"
            promo_overview[promo_type] = {
                "transaction_count": row['transaction_count'],
                "total_revenue": row['total_revenue'],
                "avg_transaction_value": row['avg_transaction_value'],
                "avg_quantity": row['avg_quantity'],
                "unique_products": row['unique_products'],
                "customer_segments": row['customer_segments']
            }
        
        # Calculate promotional lift
        promotional_lift = {}
        if "With Promotion" in promo_overview and "Without Promotion" in promo_overview:
            promo_data = promo_overview["With Promotion"]
            non_promo_data = promo_overview["Without Promotion"]
            
            promotional_lift = {
                "transaction_value_lift": (promo_data["avg_transaction_value"] / non_promo_data["avg_transaction_value"] - 1) * 100,
                "quantity_lift": (promo_data["avg_quantity"] / non_promo_data["avg_quantity"] - 1) * 100,
                "revenue_share": promo_data["total_revenue"] / (promo_data["total_revenue"] + non_promo_data["total_revenue"]) * 100
            }
        
        # Promotional impact by category
        cursor.execute("""
            SELECT 
                category,
                promotion_applied,
                COUNT(*) as transaction_count,
                SUM(total_amount) as total_revenue,
                AVG(total_amount) as avg_transaction_value
            FROM sales_transactions
            WHERE transaction_date >= ?
            GROUP BY category, promotion_applied
            ORDER BY category, promotion_applied
        """, (cutoff_date,))
        
        category_impact = {}
        for row in cursor.fetchall():
            category = row['category']
            promo_status = "promoted" if row['promotion_applied'] else "regular"
            
            if category not in category_impact:
                category_impact[category] = {}
            
            category_impact[category][promo_status] = {
                "transaction_count": row['transaction_count'],
                "total_revenue": row['total_revenue'],
                "avg_transaction_value": row['avg_transaction_value']
            }
        
        # Calculate category-specific lift
        for category, data in category_impact.items():
            if "promoted" in data and "regular" in data:
                promoted = data["promoted"]
                regular = data["regular"]
                
                data["lift_metrics"] = {
                    "value_lift_percent": (promoted["avg_transaction_value"] / regular["avg_transaction_value"] - 1) * 100,
                    "volume_lift_percent": (promoted["transaction_count"] / regular["transaction_count"] - 1) * 100,
                    "revenue_lift_percent": (promoted["total_revenue"] / regular["total_revenue"] - 1) * 100
                }
        
        # Promotional impact by customer segment
        cursor.execute("""
            SELECT 
                COALESCE(customer_segment, 'Unknown') as segment,
                promotion_applied,
                COUNT(*) as transaction_count,
                SUM(total_amount) as total_revenue,
                AVG(total_amount) as avg_transaction_value
            FROM sales_transactions
            WHERE transaction_date >= ?
            GROUP BY COALESCE(customer_segment, 'Unknown'), promotion_applied
            ORDER BY segment, promotion_applied
        """, (cutoff_date,))
        
        segment_impact = {}
        for row in cursor.fetchall():
            segment = row['segment']
            promo_status = "promoted" if row['promotion_applied'] else "regular"
            
            if segment not in segment_impact:
                segment_impact[segment] = {}
            
            segment_impact[segment][promo_status] = {
                "transaction_count": row['transaction_count'],
                "total_revenue": row['total_revenue'],
                "avg_transaction_value": row['avg_transaction_value']
            }
        
        # Calculate segment-specific promotional responsiveness
        for segment, data in segment_impact.items():
            if "promoted" in data and "regular" in data:
                promoted = data["promoted"]
                regular = data["regular"]
                
                # Calculate promotional responsiveness score
                value_response = promoted["avg_transaction_value"] / regular["avg_transaction_value"] if regular["avg_transaction_value"] > 0 else 1
                volume_response = promoted["transaction_count"] / regular["transaction_count"] if regular["transaction_count"] > 0 else 1
                
                data["responsiveness"] = {
                    "value_response_ratio": value_response,
                    "volume_response_ratio": volume_response,
                    "overall_responsiveness": (value_response + volume_response) / 2,
                    "promotion_preference": promoted["transaction_count"] / (promoted["transaction_count"] + regular["transaction_count"]) * 100
                }
        
        # Seasonal adjustment analysis
        cursor.execute("""
            SELECT 
                strftime('%m', transaction_date) as month,
                promotion_applied,
                COUNT(*) as transaction_count,
                SUM(total_amount) as total_revenue,
                AVG(total_amount) as avg_transaction_value
            FROM sales_transactions
            WHERE transaction_date >= ?
            GROUP BY strftime('%m', transaction_date), promotion_applied
            ORDER BY month, promotion_applied
        """, (cutoff_date,))
        
        seasonal_promo_impact = {}
        for row in cursor.fetchall():
            month = int(row['month'])
            promo_status = "promoted" if row['promotion_applied'] else "regular"
            
            if month not in seasonal_promo_impact:
                seasonal_promo_impact[month] = {}
            
            seasonal_promo_impact[month][promo_status] = {
                "transaction_count": row['transaction_count'],
                "total_revenue": row['total_revenue'],
                "avg_transaction_value": row['avg_transaction_value']
            }
        
        conn.close()
        
        # Generate insights and recommendations
        insights = []
        recommendations = []
        
        # Overall promotional effectiveness
        if promotional_lift:
            if promotional_lift["transaction_value_lift"] > 10:
                insights.append(f"Promotions increase transaction value by {promotional_lift['transaction_value_lift']:.1f}%")
                recommendations.append("Continue promotional strategy - showing strong value lift")
            elif promotional_lift["transaction_value_lift"] < 0:
                insights.append("Promotions may be cannibalizing regular sales")
                recommendations.append("Review promotional strategy and pricing")
        
        # Category-specific insights
        best_category = None
        best_lift = 0
        for category, data in category_impact.items():
            if "lift_metrics" in data:
                lift = data["lift_metrics"]["value_lift_percent"]
                if lift > best_lift:
                    best_lift = lift
                    best_category = category
        
        if best_category:
            insights.append(f"{best_category} shows highest promotional response ({best_lift:.1f}% lift)")
            recommendations.append(f"Focus promotional budget on {best_category} category")
        
        # Segment-specific insights
        most_responsive_segment = None
        highest_responsiveness = 0
        for segment, data in segment_impact.items():
            if "responsiveness" in data:
                responsiveness = data["responsiveness"]["overall_responsiveness"]
                if responsiveness > highest_responsiveness:
                    highest_responsiveness = responsiveness
                    most_responsive_segment = segment
        
        if most_responsive_segment:
            insights.append(f"{most_responsive_segment} segment most responsive to promotions")
            recommendations.append(f"Target {most_responsive_segment} segment with promotional campaigns")
        
        return {
            "success": True,
            "message": f"Promotional impact analysis completed for {days_back} days",
            "data": {
                "promotional_overview": promo_overview,
                "promotional_lift": promotional_lift,
                "category_impact": category_impact,
                "segment_impact": segment_impact,
                "seasonal_impact": seasonal_promo_impact,
                "insights": insights,
                "recommendations": recommendations,
                "analysis_period_days": days_back,
                "analyzed_at": datetime.now().isoformat()
            }
        }
        
    except sqlite3.Error as e:
        logging.error(f"Database error in analyze_promotional_impact: {e}")
        return {
            "success": False,
            "message": f"Database error analyzing promotional impact: {e}"
        }
    except Exception as e:
        logging.error(f"Error in analyze_promotional_impact: {e}")
        return {
            "success": False,
            "message": f"Error analyzing promotional impact: {e}"
        }


# --- MCP Server Setup ---
server = Server("sales-data-mcp-server")

@server.list_tools()
async def handle_list_tools() -> List[mcp_types.Tool]:
    """List available tools for sales data collection and analysis."""
    return [
        mcp_types.Tool(
            name="collect_sales_data",
            description="Collect sales transaction data from POS and e-commerce systems for model training",
            inputSchema={
                "type": "object",
                "properties": {
                    "days_back": {
                        "type": "integer",
                        "description": f"Number of days back to collect data (default: {DEFAULT_DATA_COLLECTION_DAYS})",
                        "default": DEFAULT_DATA_COLLECTION_DAYS
                    }
                }
            }
        ),
        mcp_types.Tool(
            name="analyze_customer_patterns",
            description="Analyze customer purchasing patterns and behavior for demand prediction context",
            inputSchema={
                "type": "object",
                "properties": {
                    "segment": {
                        "type": "string",
                        "description": "Specific customer segment to analyze (optional - analyzes all segments if not provided)"
                    }
                }
            }
        ),
        mcp_types.Tool(
            name="extract_seasonal_trends",
            description="Extract seasonal trends and cyclical patterns from sales data",
            inputSchema={
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "Specific product category to analyze (optional - analyzes all categories if not provided)"
                    }
                }
            }
        ),
        mcp_types.Tool(
            name="clean_and_transform_data",
            description="Implement automated data cleaning and transformation functions",
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset_info": {
                        "type": "object",
                        "description": "Dataset information for cleaning (optional - cleans recent data if not provided)",
                        "default": {}
                    }
                }
            }
        ),
        mcp_types.Tool(
            name="assess_data_completeness",
            description="Assess data completeness with detailed completeness checks",
            inputSchema={
                "type": "object",
                "properties": {
                    "days_back": {
                        "type": "integer",
                        "description": f"Number of days back to assess completeness (default: {DEFAULT_DATA_COLLECTION_DAYS})",
                        "default": DEFAULT_DATA_COLLECTION_DAYS
                    }
                }
            }
        ),
        mcp_types.Tool(
            name="validate_data_quality",
            description="Validate data quality with completeness and consistency checks",
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset_info": {
                        "type": "object",
                        "description": "Dataset information for validation (optional - validates recent data if not provided)",
                        "default": {}
                    }
                }
            }
        ),
        mcp_types.Tool(
            name="segment_customers_by_behavior",
            description="Implement customer segmentation algorithms based on purchase patterns using RFM analysis",
            inputSchema={
                "type": "object",
                "properties": {
                    "days_back": {
                        "type": "integer",
                        "description": f"Number of days back to analyze for segmentation (default: {DEFAULT_DATA_COLLECTION_DAYS})",
                        "default": DEFAULT_DATA_COLLECTION_DAYS
                    }
                }
            }
        ),
        mcp_types.Tool(
            name="analyze_promotional_impact",
            description="Analyze promotional impact and seasonal adjustment tools for demand prediction context",
            inputSchema={
                "type": "object",
                "properties": {
                    "days_back": {
                        "type": "integer",
                        "description": f"Number of days back to analyze promotional impact (default: {DEFAULT_DATA_COLLECTION_DAYS})",
                        "default": DEFAULT_DATA_COLLECTION_DAYS
                    }
                }
            }
        )
    ]


@server.call_tool()
async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> List[mcp_types.TextContent]:
    """Handle tool calls for sales data operations."""
    try:
        if name == "collect_sales_data":
            days_back = arguments.get("days_back", DEFAULT_DATA_COLLECTION_DAYS)
            result = collect_sales_data(days_back)
            
        elif name == "analyze_customer_patterns":
            segment = arguments.get("segment")
            result = analyze_customer_patterns(segment)
            
        elif name == "extract_seasonal_trends":
            category = arguments.get("category")
            result = extract_seasonal_trends(category)
            
        elif name == "clean_and_transform_data":
            dataset_info = arguments.get("dataset_info", {})
            result = clean_and_transform_data(dataset_info)
            
        elif name == "assess_data_completeness":
            days_back = arguments.get("days_back", DEFAULT_DATA_COLLECTION_DAYS)
            result = assess_data_completeness(days_back)
            
        elif name == "validate_data_quality":
            dataset_info = arguments.get("dataset_info", {})
            result = validate_data_quality(dataset_info)
            
        elif name == "segment_customers_by_behavior":
            days_back = arguments.get("days_back", DEFAULT_DATA_COLLECTION_DAYS)
            result = segment_customers_by_behavior(days_back)
            
        elif name == "analyze_promotional_impact":
            days_back = arguments.get("days_back", DEFAULT_DATA_COLLECTION_DAYS)
            result = analyze_promotional_impact(days_back)
            
        else:
            result = {
                "success": False,
                "message": f"Unknown tool: {name}"
            }
        
        return [mcp_types.TextContent(
            type="text",
            text=json.dumps(result, indent=2, default=str)
        )]
        
    except Exception as e:
        logging.error(f"Error handling tool call {name}: {e}")
        return [mcp_types.TextContent(
            type="text",
            text=json.dumps({
                "success": False,
                "message": f"Error executing {name}: {str(e)}"
            }, indent=2)
        )]


async def main():
    """Main server entry point."""
    # Use stdin/stdout for MCP communication
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="sales-data-mcp-server",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


if __name__ == "__main__":
    asyncio.run(main())