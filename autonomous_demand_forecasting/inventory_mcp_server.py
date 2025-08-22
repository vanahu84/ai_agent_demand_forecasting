"""
Inventory MCP Server for Autonomous Demand Forecasting System.

This server gathers inventory levels and supply chain data for demand forecasting context,
including warehouse management system integration, stock level monitoring, and availability tracking.
"""

import asyncio
import json
import logging
import os
import sqlite3
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple

import mcp.server.stdio
from dotenv import load_dotenv

# MCP Server Imports
from mcp import types as mcp_types
from mcp.server.lowlevel import NotificationOptions, Server
from mcp.server.models import InitializationOptions

# Import data models
from autonomous_demand_forecasting.database.models import (
    Product, Store
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
DEFAULT_INVENTORY_ANALYSIS_DAYS = 30
MIN_STOCK_THRESHOLD = 10
STOCKOUT_ANALYSIS_DAYS = 30
REORDER_POINT_MULTIPLIER = 1.5
FORECASTING_CONTEXT_DAYS = 90
LEAD_TIME_ANALYSIS_DAYS = 60
SEASONAL_ANALYSIS_DAYS = 365


# --- Database Utility Functions ---
def get_db_connection():
    """Get database connection with row factory."""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# --- Data Models ---
class InventorySnapshot:
    """Represents current inventory state across locations."""
    def __init__(self, product_id: str, store_id: str, current_stock: int, 
                 reserved_stock: int, available_stock: int, reorder_point: Optional[int] = None,
                 max_stock_level: Optional[int] = None, last_updated: Optional[datetime] = None):
        self.product_id = product_id
        self.store_id = store_id
        self.current_stock = current_stock
        self.reserved_stock = reserved_stock
        self.available_stock = available_stock
        self.reorder_point = reorder_point
        self.max_stock_level = max_stock_level
        self.last_updated = last_updated or datetime.now()


class StockoutEvent:
    """Represents a stockout event with impact analysis."""
    def __init__(self, product_id: str, store_id: str, stockout_date: datetime,
                 duration_hours: Optional[int] = None, lost_sales_estimate: Optional[float] = None,
                 restock_date: Optional[datetime] = None):
        self.product_id = product_id
        self.store_id = store_id
        self.stockout_date = stockout_date
        self.duration_hours = duration_hours
        self.lost_sales_estimate = lost_sales_estimate
        self.restock_date = restock_date


class StockoutAnalysis:
    """Analysis of stockout patterns and impact."""
    def __init__(self, total_events: int, affected_products: List[str], 
                 total_lost_sales: float, avg_duration_hours: float,
                 most_affected_categories: List[str]):
        self.total_events = total_events
        self.affected_products = affected_products
        self.total_lost_sales = total_lost_sales
        self.avg_duration_hours = avg_duration_hours
        self.most_affected_categories = most_affected_categories


class TurnoverMetrics:
    """Inventory turnover analysis metrics."""
    def __init__(self, product_id: str, turnover_rate: float, days_of_supply: float,
                 avg_stock_level: float, sales_velocity: float):
        self.product_id = product_id
        self.turnover_rate = turnover_rate
        self.days_of_supply = days_of_supply
        self.avg_stock_level = avg_stock_level
        self.sales_velocity = sales_velocity


class SupplyChainStatus:
    """Overall supply chain health assessment."""
    def __init__(self, overall_health_score: float, critical_stockouts: int,
                 low_stock_alerts: int, supply_chain_constraints: List[str],
                 recommendations: List[str]):
        self.overall_health_score = overall_health_score
        self.critical_stockouts = critical_stockouts
        self.low_stock_alerts = low_stock_alerts
        self.supply_chain_constraints = supply_chain_constraints
        self.recommendations = recommendations


# --- Core Inventory Functions ---
def get_current_inventory(location: Optional[str] = None) -> Dict[str, Any]:
    """
    Get current inventory snapshot across all locations or specific location.
    
    Monitors current stock levels across all locations with availability tracking.
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Base query for inventory levels
        base_query = """
            SELECT 
                il.product_id,
                il.store_id,
                il.current_stock,
                il.reserved_stock,
                il.available_stock,
                il.reorder_point,
                il.max_stock_level,
                il.last_updated,
                p.name as product_name,
                p.category,
                p.unit_price,
                s.name as store_name,
                s.location,
                s.region
            FROM inventory_levels il
            JOIN products p ON il.product_id = p.id
            JOIN stores s ON il.store_id = s.id
            WHERE p.active = 1 AND s.active = 1
        """
        
        params = []
        if location:
            base_query += " AND (s.location = ? OR s.region = ?)"
            params.extend([location, location])
        
        base_query += " ORDER BY il.last_updated DESC"
        
        cursor.execute(base_query, params)
        rows = cursor.fetchall()
        
        # Process inventory data
        inventory_items = []
        total_products = 0
        low_stock_items = 0
        out_of_stock_items = 0
        total_inventory_value = 0
        
        for row in rows:
            # Parse datetime if it's a string
            last_updated = row['last_updated']
            if isinstance(last_updated, str):
                last_updated = datetime.fromisoformat(last_updated)
            
            inventory_item = InventorySnapshot(
                product_id=row['product_id'],
                store_id=row['store_id'],
                current_stock=row['current_stock'],
                reserved_stock=row['reserved_stock'] or 0,
                available_stock=row['available_stock'],
                reorder_point=row['reorder_point'],
                max_stock_level=row['max_stock_level'],
                last_updated=last_updated
            )
            
            # Add product and store details
            item_data = {
                **inventory_item.__dict__,
                'product_name': row['product_name'],
                'category': row['category'],
                'unit_price': row['unit_price'],
                'store_name': row['store_name'],
                'location': row['location'],
                'region': row['region'],
                'inventory_value': row['current_stock'] * (row['unit_price'] or 0)
            }
            
            inventory_items.append(item_data)
            total_products += 1
            
            # Calculate summary statistics
            if row['available_stock'] <= 0:
                out_of_stock_items += 1
            elif row['reorder_point'] and row['available_stock'] <= row['reorder_point']:
                low_stock_items += 1
            
            total_inventory_value += item_data['inventory_value']
        
        # Calculate additional metrics
        categories = list(set(item['category'] for item in inventory_items))
        locations = list(set(item['location'] for item in inventory_items))
        
        # Stock level distribution
        stock_levels = [item['available_stock'] for item in inventory_items]
        avg_stock_level = statistics.mean(stock_levels) if stock_levels else 0
        
        conn.close()
        
        return {
            "success": True,
            "message": f"Retrieved inventory data for {total_products} products",
            "data": {
                "inventory_items": inventory_items,
                "summary": {
                    "total_products": total_products,
                    "total_inventory_value": total_inventory_value,
                    "out_of_stock_items": out_of_stock_items,
                    "low_stock_items": low_stock_items,
                    "avg_stock_level": avg_stock_level,
                    "categories_count": len(categories),
                    "locations_count": len(locations)
                },
                "categories": categories,
                "locations": locations,
                "retrieved_at": datetime.now().isoformat()
            }
        }
        
    except sqlite3.Error as e:
        logging.error(f"Database error in get_current_inventory: {e}")
        return {
            "success": False,
            "message": f"Database error retrieving inventory: {e}"
        }
    except Exception as e:
        logging.error(f"Error in get_current_inventory: {e}")
        return {
            "success": False,
            "message": f"Error retrieving inventory: {e}"
        }


def analyze_stockout_patterns(days_back: int = STOCKOUT_ANALYSIS_DAYS) -> Dict[str, Any]:
    """
    Analyze stockout patterns and their impact on sales.
    
    Provides comprehensive analysis of stockout events including frequency, duration, and impact.
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        # Get stockout events with product and store details
        cursor.execute("""
            SELECT 
                so.product_id,
                so.store_id,
                so.stockout_date,
                so.duration_hours,
                so.lost_sales_estimate,
                so.restock_date,
                p.name as product_name,
                p.category,
                p.unit_price,
                s.name as store_name,
                s.location,
                s.region
            FROM stockout_events so
            JOIN products p ON so.product_id = p.id
            JOIN stores s ON so.store_id = s.id
            WHERE so.stockout_date >= ?
            ORDER BY so.stockout_date DESC
        """, (cutoff_date,))
        
        rows = cursor.fetchall()
        
        if not rows:
            return {
                "success": True,
                "message": "No stockout events found in the specified period",
                "data": {
                    "stockout_events": [],
                    "analysis": {
                        "total_events": 0,
                        "total_lost_sales": 0,
                        "avg_duration_hours": 0,
                        "most_affected_categories": [],
                        "most_affected_products": []
                    }
                }
            }
        
        # Process stockout events
        stockout_events = []
        total_lost_sales = 0
        duration_hours_list = []
        category_impact = {}
        product_impact = {}
        location_impact = {}
        
        for row in rows:
            # Parse datetime if it's a string
            stockout_date = row['stockout_date']
            if isinstance(stockout_date, str):
                stockout_date = datetime.fromisoformat(stockout_date)
            
            restock_date = None
            if row['restock_date']:
                if isinstance(row['restock_date'], str):
                    restock_date = datetime.fromisoformat(row['restock_date'])
                else:
                    restock_date = row['restock_date']
            
            event_data = {
                'product_id': row['product_id'],
                'store_id': row['store_id'],
                'stockout_date': stockout_date.isoformat(),
                'duration_hours': row['duration_hours'],
                'lost_sales_estimate': row['lost_sales_estimate'] or 0,
                'restock_date': restock_date.isoformat() if restock_date else None,
                'product_name': row['product_name'],
                'category': row['category'],
                'unit_price': row['unit_price'],
                'store_name': row['store_name'],
                'location': row['location'],
                'region': row['region']
            }
            
            stockout_events.append(event_data)
            
            # Aggregate impact data
            lost_sales = row['lost_sales_estimate'] or 0
            total_lost_sales += lost_sales
            
            if row['duration_hours']:
                duration_hours_list.append(row['duration_hours'])
            
            # Category impact
            category = row['category']
            if category not in category_impact:
                category_impact[category] = {'events': 0, 'lost_sales': 0, 'total_duration': 0}
            category_impact[category]['events'] += 1
            category_impact[category]['lost_sales'] += lost_sales
            category_impact[category]['total_duration'] += row['duration_hours'] or 0
            
            # Product impact
            product_key = f"{row['product_id']}_{row['product_name']}"
            if product_key not in product_impact:
                product_impact[product_key] = {'events': 0, 'lost_sales': 0}
            product_impact[product_key]['events'] += 1
            product_impact[product_key]['lost_sales'] += lost_sales
            
            # Location impact
            location = row['location']
            if location not in location_impact:
                location_impact[location] = {'events': 0, 'lost_sales': 0}
            location_impact[location]['events'] += 1
            location_impact[location]['lost_sales'] += lost_sales
        
        # Calculate analysis metrics
        total_events = len(stockout_events)
        avg_duration_hours = statistics.mean(duration_hours_list) if duration_hours_list else 0
        
        # Most affected categories (by number of events)
        most_affected_categories = sorted(
            category_impact.items(), 
            key=lambda x: x[1]['events'], 
            reverse=True
        )[:5]
        
        # Most affected products (by lost sales)
        most_affected_products = sorted(
            product_impact.items(), 
            key=lambda x: x[1]['lost_sales'], 
            reverse=True
        )[:5]
        
        # Most affected locations
        most_affected_locations = sorted(
            location_impact.items(), 
            key=lambda x: x[1]['events'], 
            reverse=True
        )[:5]
        
        # Generate insights and recommendations
        insights = []
        recommendations = []
        
        if total_events > 0:
            insights.append(f"Total of {total_events} stockout events in the last {days_back} days")
            insights.append(f"Average stockout duration: {avg_duration_hours:.1f} hours")
            insights.append(f"Total estimated lost sales: ${total_lost_sales:,.2f}")
            
            if most_affected_categories:
                top_category = most_affected_categories[0]
                insights.append(f"Most affected category: {top_category[0]} ({top_category[1]['events']} events)")
            
            # Recommendations based on analysis
            if avg_duration_hours > 24:
                recommendations.append("Consider improving restock procedures - average stockout duration exceeds 24 hours")
            
            if total_lost_sales > 10000:
                recommendations.append("High lost sales impact - prioritize inventory optimization for top affected products")
            
            high_frequency_categories = [cat for cat, data in category_impact.items() if data['events'] > 5]
            if high_frequency_categories:
                recommendations.append(f"Increase safety stock for high-frequency stockout categories: {', '.join(high_frequency_categories)}")
        
        conn.close()
        
        return {
            "success": True,
            "message": f"Analyzed {total_events} stockout events over {days_back} days",
            "data": {
                "stockout_events": stockout_events,
                "analysis": {
                    "total_events": total_events,
                    "total_lost_sales": total_lost_sales,
                    "avg_duration_hours": avg_duration_hours,
                    "most_affected_categories": [
                        {"category": cat, "events": data['events'], "lost_sales": data['lost_sales']}
                        for cat, data in most_affected_categories
                    ],
                    "most_affected_products": [
                        {"product": prod.split('_', 1)[1], "events": data['events'], "lost_sales": data['lost_sales']}
                        for prod, data in most_affected_products
                    ],
                    "most_affected_locations": [
                        {"location": loc, "events": data['events'], "lost_sales": data['lost_sales']}
                        for loc, data in most_affected_locations
                    ]
                },
                "insights": insights,
                "recommendations": recommendations,
                "analysis_period_days": days_back,
                "analyzed_at": datetime.now().isoformat()
            }
        }
        
    except sqlite3.Error as e:
        logging.error(f"Database error in analyze_stockout_patterns: {e}")
        return {
            "success": False,
            "message": f"Database error analyzing stockout patterns: {e}"
        }
    except Exception as e:
        logging.error(f"Error in analyze_stockout_patterns: {e}")
        return {
            "success": False,
            "message": f"Error analyzing stockout patterns: {e}"
        }


def calculate_inventory_turnover(product_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Calculate inventory turnover metrics for products.
    
    Provides turnover analysis including turnover rate, days of supply, and sales velocity.
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Base query for turnover calculation
        base_query = """
            SELECT 
                p.id as product_id,
                p.name as product_name,
                p.category,
                p.unit_price,
                AVG(il.current_stock) as avg_stock_level,
                SUM(st.quantity) as total_sales_quantity,
                SUM(st.total_amount) as total_sales_value,
                COUNT(DISTINCT DATE(st.transaction_date)) as active_sales_days
            FROM products p
            LEFT JOIN inventory_levels il ON p.id = il.product_id
            LEFT JOIN sales_transactions st ON p.id = st.product_id 
                AND st.transaction_date >= ?
            WHERE p.active = 1
        """
        
        params = [datetime.now() - timedelta(days=DEFAULT_INVENTORY_ANALYSIS_DAYS)]
        
        if product_id:
            base_query += " AND p.id = ?"
            params.append(product_id)
        
        base_query += """
            GROUP BY p.id, p.name, p.category, p.unit_price
            HAVING AVG(il.current_stock) > 0
            ORDER BY total_sales_quantity DESC
        """
        
        cursor.execute(base_query, params)
        rows = cursor.fetchall()
        
        if not rows:
            return {
                "success": True,
                "message": "No inventory turnover data available",
                "data": {
                    "turnover_metrics": [],
                    "summary": {
                        "total_products_analyzed": 0,
                        "avg_turnover_rate": 0,
                        "high_turnover_products": 0,
                        "low_turnover_products": 0
                    }
                }
            }
        
        # Calculate turnover metrics
        turnover_metrics = []
        turnover_rates = []
        
        for row in rows:
            avg_stock = row['avg_stock_level'] or 0
            total_sales = row['total_sales_quantity'] or 0
            active_days = row['active_sales_days'] or 0
            
            # Calculate turnover rate (sales / average inventory)
            turnover_rate = total_sales / avg_stock if avg_stock > 0 else 0
            
            # Calculate days of supply (how many days current stock will last)
            daily_sales_rate = total_sales / DEFAULT_INVENTORY_ANALYSIS_DAYS if total_sales > 0 else 0
            days_of_supply = avg_stock / daily_sales_rate if daily_sales_rate > 0 else float('inf')
            
            # Sales velocity (units sold per active day)
            sales_velocity = total_sales / active_days if active_days > 0 else 0
            
            # Classify turnover performance
            if turnover_rate > 12:  # More than 12 times per analysis period
                turnover_classification = "High"
            elif turnover_rate > 4:  # 4-12 times per analysis period
                turnover_classification = "Medium"
            elif turnover_rate > 1:  # 1-4 times per analysis period
                turnover_classification = "Low"
            else:
                turnover_classification = "Very Low"
            
            metrics = {
                'product_id': row['product_id'],
                'product_name': row['product_name'],
                'category': row['category'],
                'unit_price': row['unit_price'],
                'avg_stock_level': avg_stock,
                'total_sales_quantity': total_sales,
                'total_sales_value': row['total_sales_value'] or 0,
                'turnover_rate': turnover_rate,
                'days_of_supply': min(days_of_supply, 999),  # Cap at 999 for display
                'sales_velocity': sales_velocity,
                'active_sales_days': active_days,
                'turnover_classification': turnover_classification
            }
            
            turnover_metrics.append(metrics)
            if turnover_rate > 0:
                turnover_rates.append(turnover_rate)
        
        # Calculate summary statistics
        total_products = len(turnover_metrics)
        avg_turnover_rate = statistics.mean(turnover_rates) if turnover_rates else 0
        high_turnover_products = len([m for m in turnover_metrics if m['turnover_classification'] in ['High', 'Medium']])
        low_turnover_products = len([m for m in turnover_metrics if m['turnover_classification'] in ['Low', 'Very Low']])
        
        # Category analysis
        category_turnover = {}
        for metric in turnover_metrics:
            category = metric['category']
            if category not in category_turnover:
                category_turnover[category] = {'products': 0, 'total_turnover': 0, 'avg_turnover': 0}
            category_turnover[category]['products'] += 1
            category_turnover[category]['total_turnover'] += metric['turnover_rate']
        
        for category, data in category_turnover.items():
            data['avg_turnover'] = data['total_turnover'] / data['products'] if data['products'] > 0 else 0
        
        # Generate recommendations
        recommendations = []
        
        very_low_turnover = [m for m in turnover_metrics if m['turnover_classification'] == 'Very Low']
        if very_low_turnover:
            recommendations.append(f"Consider reducing stock levels for {len(very_low_turnover)} very low turnover products")
        
        high_turnover = [m for m in turnover_metrics if m['turnover_classification'] == 'High']
        if high_turnover:
            recommendations.append(f"Consider increasing stock levels for {len(high_turnover)} high turnover products to avoid stockouts")
        
        # Products with very high days of supply
        excess_inventory = [m for m in turnover_metrics if m['days_of_supply'] > 90]
        if excess_inventory:
            recommendations.append(f"Review {len(excess_inventory)} products with >90 days of supply for potential overstock")
        
        conn.close()
        
        return {
            "success": True,
            "message": f"Calculated turnover metrics for {total_products} products",
            "data": {
                "turnover_metrics": turnover_metrics,
                "summary": {
                    "total_products_analyzed": total_products,
                    "avg_turnover_rate": avg_turnover_rate,
                    "high_turnover_products": high_turnover_products,
                    "low_turnover_products": low_turnover_products
                },
                "category_analysis": category_turnover,
                "recommendations": recommendations,
                "analysis_period_days": DEFAULT_INVENTORY_ANALYSIS_DAYS,
                "analyzed_at": datetime.now().isoformat()
            }
        }
        
    except sqlite3.Error as e:
        logging.error(f"Database error in calculate_inventory_turnover: {e}")
        return {
            "success": False,
            "message": f"Database error calculating inventory turnover: {e}"
        }
    except Exception as e:
        logging.error(f"Error in calculate_inventory_turnover: {e}")
        return {
            "success": False,
            "message": f"Error calculating inventory turnover: {e}"
        }


def assess_supply_chain_health() -> Dict[str, Any]:
    """
    Assess overall supply chain health and identify constraints.
    
    Provides comprehensive supply chain health assessment with recommendations.
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get current inventory status
        cursor.execute("""
            SELECT 
                COUNT(*) as total_products,
                COUNT(CASE WHEN il.available_stock <= 0 THEN 1 END) as out_of_stock,
                COUNT(CASE WHEN il.reorder_point IS NOT NULL AND il.available_stock <= il.reorder_point THEN 1 END) as below_reorder_point,
                COUNT(CASE WHEN il.available_stock > COALESCE(il.max_stock_level, 1000) THEN 1 END) as overstocked,
                AVG(il.available_stock) as avg_stock_level
            FROM inventory_levels il
            JOIN products p ON il.product_id = p.id
            WHERE p.active = 1
        """)
        
        inventory_status = cursor.fetchone()
        
        # Get recent stockout events
        recent_cutoff = datetime.now() - timedelta(days=7)
        cursor.execute("""
            SELECT COUNT(*) as recent_stockouts,
                   AVG(duration_hours) as avg_stockout_duration,
                   SUM(lost_sales_estimate) as total_lost_sales
            FROM stockout_events
            WHERE stockout_date >= ?
        """, (recent_cutoff,))
        
        stockout_status = cursor.fetchone()
        
        # Get inventory turnover issues
        cursor.execute("""
            SELECT 
                p.category,
                COUNT(*) as products_in_category,
                AVG(il.available_stock) as avg_stock,
                COUNT(CASE WHEN il.available_stock <= 0 THEN 1 END) as stockouts_in_category
            FROM inventory_levels il
            JOIN products p ON il.product_id = p.id
            WHERE p.active = 1
            GROUP BY p.category
            ORDER BY stockouts_in_category DESC, avg_stock ASC
        """)
        
        category_analysis = cursor.fetchall()
        
        # Calculate health scores
        total_products = inventory_status['total_products'] or 1
        out_of_stock_rate = (inventory_status['out_of_stock'] or 0) / total_products
        reorder_alert_rate = (inventory_status['below_reorder_point'] or 0) / total_products
        overstock_rate = (inventory_status['overstocked'] or 0) / total_products
        
        # Overall health score (0-100)
        health_score = 100
        health_score -= out_of_stock_rate * 40  # Heavy penalty for stockouts
        health_score -= reorder_alert_rate * 20  # Moderate penalty for low stock
        health_score -= overstock_rate * 10     # Light penalty for overstock
        
        # Recent stockout impact
        recent_stockouts = stockout_status['recent_stockouts'] or 0
        if recent_stockouts > 0:
            health_score -= min(recent_stockouts * 2, 20)  # Up to 20 point penalty
        
        health_score = max(health_score, 0)  # Don't go below 0
        
        # Identify constraints and issues
        constraints = []
        recommendations = []
        
        # Stock level constraints
        if out_of_stock_rate > 0.05:  # More than 5% out of stock
            constraints.append(f"High stockout rate: {out_of_stock_rate:.1%} of products out of stock")
            recommendations.append("Implement automated reordering for critical products")
        
        if reorder_alert_rate > 0.15:  # More than 15% below reorder point
            constraints.append(f"Low stock alerts: {reorder_alert_rate:.1%} of products below reorder point")
            recommendations.append("Review and adjust reorder points based on demand patterns")
        
        if overstock_rate > 0.10:  # More than 10% overstocked
            constraints.append(f"Overstock issues: {overstock_rate:.1%} of products overstocked")
            recommendations.append("Implement inventory optimization to reduce excess stock")
        
        # Recent performance constraints
        if recent_stockouts > 5:
            constraints.append(f"Recent stockout spike: {recent_stockouts} stockouts in the last 7 days")
            recommendations.append("Investigate supply chain disruptions and improve demand forecasting")
        
        avg_stockout_duration = stockout_status['avg_stockout_duration'] or 0
        if avg_stockout_duration > 48:  # More than 48 hours average
            constraints.append(f"Long stockout duration: {avg_stockout_duration:.1f} hours average")
            recommendations.append("Improve supplier response times and emergency procurement procedures")
        
        # Category-specific constraints
        problematic_categories = []
        for row in category_analysis:
            category = row['category']
            products_count = row['products_in_category']
            stockouts = row['stockouts_in_category']
            
            if products_count > 0 and (stockouts / products_count) > 0.20:  # More than 20% stockout rate
                problematic_categories.append(category)
                constraints.append(f"Category supply issues: {category} has high stockout rate")
        
        if problematic_categories:
            recommendations.append(f"Focus supply chain improvements on categories: {', '.join(problematic_categories)}")
        
        # Generate health classification
        if health_score >= 90:
            health_classification = "Excellent"
        elif health_score >= 75:
            health_classification = "Good"
        elif health_score >= 60:
            health_classification = "Fair"
        elif health_score >= 40:
            health_classification = "Poor"
        else:
            health_classification = "Critical"
        
        # Add general recommendations based on health score
        if health_score < 60:
            recommendations.append("Immediate supply chain optimization required")
            recommendations.append("Consider implementing advanced demand forecasting")
        elif health_score < 80:
            recommendations.append("Monitor supply chain performance closely")
            recommendations.append("Optimize safety stock levels")
        else:
            recommendations.append("Maintain current supply chain practices")
            recommendations.append("Continue monitoring for continuous improvement")
        
        conn.close()
        
        return {
            "success": True,
            "message": f"Supply chain health assessment completed - {health_classification} status",
            "data": {
                "overall_health_score": round(health_score, 1),
                "health_classification": health_classification,
                "inventory_status": {
                    "total_products": total_products,
                    "out_of_stock_count": inventory_status['out_of_stock'] or 0,
                    "out_of_stock_rate": round(out_of_stock_rate * 100, 1),
                    "low_stock_alerts": inventory_status['below_reorder_point'] or 0,
                    "low_stock_rate": round(reorder_alert_rate * 100, 1),
                    "overstocked_count": inventory_status['overstocked'] or 0,
                    "overstock_rate": round(overstock_rate * 100, 1),
                    "avg_stock_level": round(inventory_status['avg_stock_level'] or 0, 1)
                },
                "recent_performance": {
                    "recent_stockouts": recent_stockouts,
                    "avg_stockout_duration_hours": round(avg_stockout_duration, 1),
                    "total_lost_sales": stockout_status['total_lost_sales'] or 0
                },
                "category_analysis": [
                    {
                        "category": row['category'],
                        "products_count": row['products_in_category'],
                        "avg_stock": round(row['avg_stock'] or 0, 1),
                        "stockouts": row['stockouts_in_category'],
                        "stockout_rate": round((row['stockouts_in_category'] / row['products_in_category']) * 100, 1) if row['products_in_category'] > 0 else 0
                    }
                    for row in category_analysis
                ],
                "supply_chain_constraints": constraints,
                "recommendations": recommendations,
                "assessed_at": datetime.now().isoformat()
            }
        }
        
    except sqlite3.Error as e:
        logging.error(f"Database error in assess_supply_chain_health: {e}")
        return {
            "success": False,
            "message": f"Database error assessing supply chain health: {e}"
        }
    except Exception as e:
        logging.error(f"Error in assess_supply_chain_health: {e}")
        return {
            "success": False,
            "message": f"Error assessing supply chain health: {e}"
        }


def detect_stockout_events(product_id: Optional[str] = None, store_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Detect current and potential stockout events with impact analysis.
    
    Identifies products at risk of stockout and analyzes potential business impact.
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Base query for stockout detection
        base_query = """
            SELECT 
                il.product_id,
                il.store_id,
                il.current_stock,
                il.available_stock,
                il.reorder_point,
                il.last_updated,
                p.name as product_name,
                p.category,
                p.unit_price,
                s.name as store_name,
                s.location,
                s.region,
                -- Calculate recent sales velocity
                COALESCE(recent_sales.daily_velocity, 0) as daily_sales_velocity,
                COALESCE(recent_sales.total_recent_sales, 0) as recent_sales_quantity
            FROM inventory_levels il
            JOIN products p ON il.product_id = p.id
            JOIN stores s ON il.store_id = s.id
            LEFT JOIN (
                SELECT 
                    st.product_id,
                    st.store_id,
                    AVG(daily_sales.daily_quantity) as daily_velocity,
                    SUM(st.quantity) as total_recent_sales
                FROM sales_transactions st
                JOIN (
                    SELECT 
                        product_id,
                        store_id,
                        DATE(transaction_date) as sale_date,
                        SUM(quantity) as daily_quantity
                    FROM sales_transactions
                    WHERE transaction_date >= ?
                    GROUP BY product_id, store_id, DATE(transaction_date)
                ) daily_sales ON st.product_id = daily_sales.product_id 
                    AND st.store_id = daily_sales.store_id
                WHERE st.transaction_date >= ?
                GROUP BY st.product_id, st.store_id
            ) recent_sales ON il.product_id = recent_sales.product_id 
                AND il.store_id = recent_sales.store_id
            WHERE p.active = 1 AND s.active = 1
        """
        
        # Parameters for recent sales analysis (last 14 days)
        recent_cutoff = datetime.now() - timedelta(days=14)
        params = [recent_cutoff, recent_cutoff]
        
        if product_id:
            base_query += " AND il.product_id = ?"
            params.append(product_id)
        
        if store_id:
            base_query += " AND il.store_id = ?"
            params.append(store_id)
        
        base_query += " ORDER BY il.available_stock ASC, recent_sales.daily_velocity DESC"
        
        cursor.execute(base_query, params)
        rows = cursor.fetchall()
        
        # Analyze stockout risks
        current_stockouts = []
        imminent_stockouts = []
        potential_stockouts = []
        
        for row in rows:
            available_stock = row['available_stock']
            daily_velocity = row['daily_sales_velocity'] or 0
            reorder_point = row['reorder_point'] or 0
            
            # Calculate days until stockout
            days_until_stockout = float('inf')
            if daily_velocity > 0:
                days_until_stockout = available_stock / daily_velocity
            
            # Estimate potential lost sales
            unit_price = row['unit_price'] or 0
            potential_daily_loss = daily_velocity * unit_price
            
            stockout_info = {
                'product_id': row['product_id'],
                'store_id': row['store_id'],
                'product_name': row['product_name'],
                'category': row['category'],
                'store_name': row['store_name'],
                'location': row['location'],
                'region': row['region'],
                'current_stock': row['current_stock'],
                'available_stock': available_stock,
                'reorder_point': reorder_point,
                'daily_sales_velocity': daily_velocity,
                'days_until_stockout': min(days_until_stockout, 999),
                'potential_daily_loss': potential_daily_loss,
                'unit_price': unit_price,
                'last_updated': row['last_updated']
            }
            
            # Classify stockout risk
            if available_stock <= 0:
                stockout_info['risk_level'] = 'CRITICAL'
                stockout_info['risk_description'] = 'Currently out of stock'
                current_stockouts.append(stockout_info)
            elif days_until_stockout <= 3:
                stockout_info['risk_level'] = 'HIGH'
                stockout_info['risk_description'] = f'Will stock out in {days_until_stockout:.1f} days'
                imminent_stockouts.append(stockout_info)
            elif days_until_stockout <= 7 or (reorder_point > 0 and available_stock <= reorder_point):
                stockout_info['risk_level'] = 'MEDIUM'
                if days_until_stockout <= 7:
                    stockout_info['risk_description'] = f'Will stock out in {days_until_stockout:.1f} days'
                else:
                    stockout_info['risk_description'] = 'Below reorder point'
                potential_stockouts.append(stockout_info)
        
        # Calculate impact summary
        total_at_risk = len(current_stockouts) + len(imminent_stockouts) + len(potential_stockouts)
        total_potential_daily_loss = sum(
            item['potential_daily_loss'] for item in current_stockouts + imminent_stockouts + potential_stockouts
        )
        
        # Category impact analysis
        category_impact = {}
        for item in current_stockouts + imminent_stockouts + potential_stockouts:
            category = item['category']
            if category not in category_impact:
                category_impact[category] = {
                    'at_risk_products': 0,
                    'potential_daily_loss': 0,
                    'critical_count': 0,
                    'high_risk_count': 0,
                    'medium_risk_count': 0
                }
            
            category_impact[category]['at_risk_products'] += 1
            category_impact[category]['potential_daily_loss'] += item['potential_daily_loss']
            
            if item['risk_level'] == 'CRITICAL':
                category_impact[category]['critical_count'] += 1
            elif item['risk_level'] == 'HIGH':
                category_impact[category]['high_risk_count'] += 1
            elif item['risk_level'] == 'MEDIUM':
                category_impact[category]['medium_risk_count'] += 1
        
        # Generate recommendations
        recommendations = []
        
        if len(current_stockouts) > 0:
            recommendations.append(f"URGENT: {len(current_stockouts)} products currently out of stock - immediate restocking required")
        
        if len(imminent_stockouts) > 0:
            recommendations.append(f"HIGH PRIORITY: {len(imminent_stockouts)} products will stock out within 3 days")
        
        if len(potential_stockouts) > 0:
            recommendations.append(f"MONITOR: {len(potential_stockouts)} products at medium risk of stockout")
        
        if total_potential_daily_loss > 1000:
            recommendations.append(f"High financial impact: ${total_potential_daily_loss:,.2f} potential daily sales loss")
        
        # Top categories at risk
        top_risk_categories = sorted(
            category_impact.items(),
            key=lambda x: x[1]['critical_count'] + x[1]['high_risk_count'],
            reverse=True
        )[:3]
        
        if top_risk_categories:
            top_categories = [cat[0] for cat in top_risk_categories]
            recommendations.append(f"Focus on categories: {', '.join(top_categories)}")
        
        conn.close()
        
        return {
            "success": True,
            "message": f"Detected {total_at_risk} products at risk of stockout",
            "data": {
                "stockout_analysis": {
                    "current_stockouts": current_stockouts,
                    "imminent_stockouts": imminent_stockouts,
                    "potential_stockouts": potential_stockouts,
                    "total_at_risk": total_at_risk,
                    "total_potential_daily_loss": total_potential_daily_loss
                },
                "category_impact": category_impact,
                "recommendations": recommendations,
                "analyzed_at": datetime.now().isoformat()
            }
        }
        
    except sqlite3.Error as e:
        logging.error(f"Database error in detect_stockout_events: {e}")
        return {
            "success": False,
            "message": f"Database error detecting stockout events: {e}"
        }
    except Exception as e:
        logging.error(f"Error in detect_stockout_events: {e}")
        return {
            "success": False,
            "message": f"Error detecting stockout events: {e}"
        }


def monitor_supply_chain_constraints() -> Dict[str, Any]:
    """
    Monitor supply chain health and identify operational constraints.
    
    Analyzes supply chain performance across multiple dimensions to identify bottlenecks.
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Analyze supplier performance (using stockout frequency as proxy)
        cursor.execute("""
            SELECT 
                p.category,
                COUNT(DISTINCT p.id) as total_products,
                COUNT(DISTINCT so.product_id) as products_with_stockouts,
                COUNT(so.id) as total_stockout_events,
                AVG(so.duration_hours) as avg_stockout_duration,
                SUM(so.lost_sales_estimate) as total_lost_sales,
                MIN(so.stockout_date) as first_stockout,
                MAX(so.stockout_date) as latest_stockout
            FROM products p
            LEFT JOIN stockout_events so ON p.id = so.product_id
                AND so.stockout_date >= ?
            WHERE p.active = 1
            GROUP BY p.category
            ORDER BY total_stockout_events DESC, total_lost_sales DESC
        """, (datetime.now() - timedelta(days=90),))
        
        category_performance = cursor.fetchall()
        
        # Analyze location-based constraints
        cursor.execute("""
            SELECT 
                s.location,
                s.region,
                COUNT(DISTINCT il.product_id) as total_products,
                COUNT(CASE WHEN il.available_stock <= 0 THEN 1 END) as out_of_stock_count,
                COUNT(CASE WHEN il.reorder_point IS NOT NULL AND il.available_stock <= il.reorder_point THEN 1 END) as low_stock_count,
                AVG(il.available_stock) as avg_stock_level,
                COUNT(DISTINCT so.product_id) as products_with_recent_stockouts,
                COUNT(so.id) as recent_stockout_events
            FROM stores s
            JOIN inventory_levels il ON s.id = il.store_id
            LEFT JOIN stockout_events so ON s.id = so.store_id
                AND so.stockout_date >= ?
            WHERE s.active = 1
            GROUP BY s.location, s.region
            ORDER BY recent_stockout_events DESC, out_of_stock_count DESC
        """, (datetime.now() - timedelta(days=30),))
        
        location_performance = cursor.fetchall()
        
        # Analyze inventory velocity and turnover constraints
        cursor.execute("""
            SELECT 
                p.category,
                AVG(il.available_stock) as avg_inventory_level,
                COUNT(CASE WHEN il.available_stock > COALESCE(il.max_stock_level, 1000) THEN 1 END) as overstocked_products,
                COUNT(CASE WHEN il.available_stock <= COALESCE(il.reorder_point, 0) THEN 1 END) as understocked_products,
                -- Calculate approximate turnover using recent sales
                COALESCE(AVG(recent_sales.sales_velocity), 0) as avg_sales_velocity,
                COUNT(DISTINCT p.id) as total_products
            FROM products p
            JOIN inventory_levels il ON p.id = il.product_id
            LEFT JOIN (
                SELECT 
                    st.product_id,
                    SUM(st.quantity) / 30.0 as sales_velocity
                FROM sales_transactions st
                WHERE st.transaction_date >= ?
                GROUP BY st.product_id
            ) recent_sales ON p.id = recent_sales.product_id
            WHERE p.active = 1
            GROUP BY p.category
            ORDER BY avg_sales_velocity DESC
        """, (datetime.now() - timedelta(days=30),))
        
        inventory_velocity = cursor.fetchall()
        
        # Identify specific constraints
        constraints = []
        constraint_details = {}
        
        # Category-based constraints
        for row in category_performance:
            category = row['category']
            total_products = row['total_products']
            stockout_events = row['total_stockout_events'] or 0
            products_with_stockouts = row['products_with_stockouts'] or 0
            
            if total_products > 0:
                stockout_rate = products_with_stockouts / total_products
                
                if stockout_rate > 0.3:  # More than 30% of products had stockouts
                    constraint_type = "High Stockout Rate"
                    constraints.append(f"{category}: {stockout_rate:.1%} of products experienced stockouts")
                    constraint_details[f"{category}_stockout"] = {
                        "type": constraint_type,
                        "severity": "HIGH" if stockout_rate > 0.5 else "MEDIUM",
                        "affected_products": products_with_stockouts,
                        "total_products": total_products,
                        "stockout_rate": stockout_rate,
                        "total_events": stockout_events,
                        "avg_duration_hours": row['avg_stockout_duration'] or 0,
                        "total_lost_sales": row['total_lost_sales'] or 0
                    }
        
        # Location-based constraints
        for row in location_performance:
            location = row['location']
            total_products = row['total_products']
            out_of_stock = row['out_of_stock_count'] or 0
            recent_stockouts = row['recent_stockout_events'] or 0
            
            if total_products > 0:
                out_of_stock_rate = out_of_stock / total_products
                
                if out_of_stock_rate > 0.1:  # More than 10% out of stock
                    constraints.append(f"{location}: {out_of_stock_rate:.1%} of products currently out of stock")
                    constraint_details[f"{location}_outofstock"] = {
                        "type": "High Out-of-Stock Rate",
                        "severity": "HIGH" if out_of_stock_rate > 0.2 else "MEDIUM",
                        "location": location,
                        "region": row['region'],
                        "out_of_stock_count": out_of_stock,
                        "out_of_stock_rate": out_of_stock_rate,
                        "recent_stockouts": recent_stockouts,
                        "avg_stock_level": row['avg_stock_level'] or 0
                    }
        
        # Inventory velocity constraints
        for row in inventory_velocity:
            category = row['category']
            overstocked = row['overstocked_products'] or 0
            understocked = row['understocked_products'] or 0
            total_products = row['total_products']
            avg_velocity = row['avg_sales_velocity'] or 0
            
            if total_products > 0:
                overstock_rate = overstocked / total_products
                understock_rate = understocked / total_products
                
                if overstock_rate > 0.2:  # More than 20% overstocked
                    constraints.append(f"{category}: {overstock_rate:.1%} of products overstocked")
                    constraint_details[f"{category}_overstock"] = {
                        "type": "Excess Inventory",
                        "severity": "MEDIUM",
                        "category": category,
                        "overstocked_products": overstocked,
                        "overstock_rate": overstock_rate,
                        "avg_inventory_level": row['avg_inventory_level'] or 0,
                        "avg_sales_velocity": avg_velocity
                    }
                
                if avg_velocity < 0.5 and understock_rate > 0.15:  # Low velocity + high understock
                    constraints.append(f"{category}: Poor demand forecasting - low velocity with high understock rate")
                    constraint_details[f"{category}_forecasting"] = {
                        "type": "Demand Forecasting Issue",
                        "severity": "MEDIUM",
                        "category": category,
                        "understocked_products": understocked,
                        "understock_rate": understock_rate,
                        "avg_sales_velocity": avg_velocity
                    }
        
        # Generate optimization recommendations
        recommendations = []
        
        # High-level recommendations based on constraint patterns
        high_severity_constraints = [k for k, v in constraint_details.items() if v.get('severity') == 'HIGH']
        
        if len(high_severity_constraints) > 0:
            recommendations.append("CRITICAL: Multiple high-severity supply chain constraints detected - immediate action required")
        
        # Category-specific recommendations
        stockout_categories = [v['category'] if 'category' in v else v.get('location', 'Unknown') 
                             for v in constraint_details.values() 
                             if v['type'] in ['High Stockout Rate', 'High Out-of-Stock Rate']]
        
        if stockout_categories:
            recommendations.append(f"Improve supplier relationships and safety stock for: {', '.join(set(stockout_categories))}")
        
        # Inventory optimization recommendations
        overstock_categories = [v['category'] for v in constraint_details.values() if v['type'] == 'Excess Inventory']
        if overstock_categories:
            recommendations.append(f"Implement inventory optimization for overstocked categories: {', '.join(overstock_categories)}")
        
        forecasting_issues = [v['category'] for v in constraint_details.values() if v['type'] == 'Demand Forecasting Issue']
        if forecasting_issues:
            recommendations.append(f"Improve demand forecasting accuracy for: {', '.join(forecasting_issues)}")
        
        # General recommendations
        if len(constraints) == 0:
            recommendations.append("Supply chain operating within normal parameters")
            recommendations.append("Continue monitoring for continuous improvement opportunities")
        elif len(constraints) < 3:
            recommendations.append("Minor supply chain optimization opportunities identified")
            recommendations.append("Focus on specific constraint areas for improvement")
        else:
            recommendations.append("Multiple supply chain constraints require systematic optimization")
            recommendations.append("Consider comprehensive supply chain review and process improvements")
        
        # Calculate overall constraint severity score
        severity_score = 0
        for constraint in constraint_details.values():
            if constraint['severity'] == 'HIGH':
                severity_score += 3
            elif constraint['severity'] == 'MEDIUM':
                severity_score += 2
            else:
                severity_score += 1
        
        # Normalize to 0-100 scale (higher = more constraints)
        max_possible_score = len(constraint_details) * 3
        constraint_severity_percentage = (severity_score / max_possible_score * 100) if max_possible_score > 0 else 0
        
        conn.close()
        
        return {
            "success": True,
            "message": f"Identified {len(constraints)} supply chain constraints",
            "data": {
                "constraint_summary": {
                    "total_constraints": len(constraints),
                    "high_severity": len([c for c in constraint_details.values() if c.get('severity') == 'HIGH']),
                    "medium_severity": len([c for c in constraint_details.values() if c.get('severity') == 'MEDIUM']),
                    "constraint_severity_score": round(constraint_severity_percentage, 1)
                },
                "constraints": constraints,
                "constraint_details": constraint_details,
                "category_performance": [
                    {
                        "category": row['category'],
                        "total_products": row['total_products'],
                        "products_with_stockouts": row['products_with_stockouts'] or 0,
                        "stockout_events": row['total_stockout_events'] or 0,
                        "avg_stockout_duration": row['avg_stockout_duration'] or 0,
                        "total_lost_sales": row['total_lost_sales'] or 0
                    }
                    for row in category_performance
                ],
                "location_performance": [
                    {
                        "location": row['location'],
                        "region": row['region'],
                        "total_products": row['total_products'],
                        "out_of_stock_count": row['out_of_stock_count'] or 0,
                        "low_stock_count": row['low_stock_count'] or 0,
                        "recent_stockout_events": row['recent_stockout_events'] or 0,
                        "avg_stock_level": round(row['avg_stock_level'] or 0, 1)
                    }
                    for row in location_performance
                ],
                "recommendations": recommendations,
                "analyzed_at": datetime.now().isoformat()
            }
        }
        
    except sqlite3.Error as e:
        logging.error(f"Database error in monitor_supply_chain_constraints: {e}")
        return {
            "success": False,
            "message": f"Database error monitoring supply chain constraints: {e}"
        }
    except Exception as e:
        logging.error(f"Error in monitor_supply_chain_constraints: {e}")
        return {
            "success": False,
            "message": f"Error monitoring supply chain constraints: {e}"
        }


def generate_inventory_optimization_recommendations(category: Optional[str] = None) -> Dict[str, Any]:
    """
    Generate inventory optimization recommendations based on turnover and stockout analysis.
    
    Provides actionable recommendations for inventory level optimization.
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get comprehensive inventory and sales data
        base_query = """
            SELECT 
                p.id as product_id,
                p.name as product_name,
                p.category,
                p.unit_price,
                il.store_id,
                s.name as store_name,
                s.location,
                il.current_stock,
                il.available_stock,
                il.reorder_point,
                il.max_stock_level,
                -- Recent sales data
                COALESCE(recent_sales.total_quantity, 0) as recent_sales_quantity,
                COALESCE(recent_sales.avg_daily_sales, 0) as avg_daily_sales,
                COALESCE(recent_sales.sales_days, 0) as active_sales_days,
                -- Stockout history
                COALESCE(stockout_history.stockout_count, 0) as recent_stockout_count,
                COALESCE(stockout_history.total_lost_sales, 0) as total_lost_sales
            FROM products p
            JOIN inventory_levels il ON p.id = il.product_id
            JOIN stores s ON il.store_id = s.id
            LEFT JOIN (
                SELECT 
                    st.product_id,
                    st.store_id,
                    SUM(st.quantity) as total_quantity,
                    AVG(daily_sales.daily_quantity) as avg_daily_sales,
                    COUNT(DISTINCT DATE(st.transaction_date)) as sales_days
                FROM sales_transactions st
                JOIN (
                    SELECT 
                        product_id,
                        store_id,
                        DATE(transaction_date) as sale_date,
                        SUM(quantity) as daily_quantity
                    FROM sales_transactions
                    WHERE transaction_date >= ?
                    GROUP BY product_id, store_id, DATE(transaction_date)
                ) daily_sales ON st.product_id = daily_sales.product_id 
                    AND st.store_id = daily_sales.store_id
                WHERE st.transaction_date >= ?
                GROUP BY st.product_id, st.store_id
            ) recent_sales ON p.id = recent_sales.product_id AND il.store_id = recent_sales.store_id
            LEFT JOIN (
                SELECT 
                    so.product_id,
                    so.store_id,
                    COUNT(*) as stockout_count,
                    SUM(so.lost_sales_estimate) as total_lost_sales
                FROM stockout_events so
                WHERE so.stockout_date >= ?
                GROUP BY so.product_id, so.store_id
            ) stockout_history ON p.id = stockout_history.product_id AND il.store_id = stockout_history.store_id
            WHERE p.active = 1 AND s.active = 1
        """
        
        # Analysis period: last 30 days for sales, 90 days for stockouts
        sales_cutoff = datetime.now() - timedelta(days=30)
        stockout_cutoff = datetime.now() - timedelta(days=90)
        params = [sales_cutoff, sales_cutoff, stockout_cutoff]
        
        if category:
            base_query += " AND p.category = ?"
            params.append(category)
        
        base_query += " ORDER BY p.category, recent_sales.avg_daily_sales DESC"
        
        cursor.execute(base_query, params)
        rows = cursor.fetchall()
        
        if not rows:
            return {
                "success": True,
                "message": "No inventory data available for optimization analysis",
                "data": {
                    "recommendations": [],
                    "optimization_opportunities": []
                }
            }
        
        # Analyze each product-store combination
        recommendations = []
        optimization_opportunities = []
        
        for row in rows:
            product_id = row['product_id']
            store_id = row['store_id']
            current_stock = row['current_stock']
            available_stock = row['available_stock']
            reorder_point = row['reorder_point'] or 0
            max_stock_level = row['max_stock_level'] or 0
            avg_daily_sales = row['avg_daily_sales'] or 0
            recent_stockouts = row['recent_stockout_count'] or 0
            lost_sales = row['total_lost_sales'] or 0
            unit_price = row['unit_price'] or 0
            
            # Calculate key metrics
            days_of_supply = available_stock / avg_daily_sales if avg_daily_sales > 0 else float('inf')
            turnover_rate = (row['recent_sales_quantity'] or 0) / current_stock if current_stock > 0 else 0
            
            # Identify optimization opportunities
            opportunity = {
                'product_id': product_id,
                'product_name': row['product_name'],
                'category': row['category'],
                'store_id': store_id,
                'store_name': row['store_name'],
                'location': row['location'],
                'current_metrics': {
                    'current_stock': current_stock,
                    'available_stock': available_stock,
                    'reorder_point': reorder_point,
                    'max_stock_level': max_stock_level,
                    'days_of_supply': min(days_of_supply, 999),
                    'avg_daily_sales': avg_daily_sales,
                    'turnover_rate': turnover_rate,
                    'recent_stockouts': recent_stockouts,
                    'lost_sales': lost_sales
                },
                'recommendations': [],
                'priority': 'LOW',
                'potential_impact': 0
            }
            
            # Stockout prevention recommendations
            if recent_stockouts > 0:
                if avg_daily_sales > 0:
                    # Suggest safety stock based on sales velocity and lead time
                    suggested_safety_stock = int(avg_daily_sales * 7)  # 7 days safety stock
                    new_reorder_point = max(reorder_point, suggested_safety_stock)
                    
                    opportunity['recommendations'].append({
                        'type': 'INCREASE_SAFETY_STOCK',
                        'description': f'Increase reorder point from {reorder_point} to {new_reorder_point} to prevent stockouts',
                        'current_value': reorder_point,
                        'suggested_value': new_reorder_point,
                        'reason': f'Had {recent_stockouts} stockouts with ${lost_sales:,.2f} lost sales'
                    })
                    opportunity['priority'] = 'HIGH'
                    opportunity['potential_impact'] = lost_sales
            
            # Overstock reduction recommendations
            if days_of_supply > 60 and turnover_rate < 2:  # More than 60 days supply with low turnover
                if max_stock_level > 0:
                    suggested_max = int(current_stock * 0.7)  # Reduce by 30%
                    opportunity['recommendations'].append({
                        'type': 'REDUCE_MAX_STOCK',
                        'description': f'Reduce maximum stock level from {max_stock_level} to {suggested_max} to free up capital',
                        'current_value': max_stock_level,
                        'suggested_value': suggested_max,
                        'reason': f'{days_of_supply:.1f} days of supply indicates overstock'
                    })
                    
                    # Calculate potential capital freed up
                    capital_freed = (current_stock - suggested_max) * unit_price
                    opportunity['potential_impact'] = max(opportunity['potential_impact'], capital_freed)
                    
                    if opportunity['priority'] == 'LOW':
                        opportunity['priority'] = 'MEDIUM'
            
            # Reorder point optimization
            if avg_daily_sales > 0 and reorder_point > 0:
                # Calculate optimal reorder point (lead time + safety stock)
                # Assuming 7-day lead time and 3-day safety stock
                optimal_reorder_point = int(avg_daily_sales * 10)
                
                if abs(reorder_point - optimal_reorder_point) > (avg_daily_sales * 2):  # Significant difference
                    opportunity['recommendations'].append({
                        'type': 'OPTIMIZE_REORDER_POINT',
                        'description': f'Adjust reorder point from {reorder_point} to {optimal_reorder_point} based on sales velocity',
                        'current_value': reorder_point,
                        'suggested_value': optimal_reorder_point,
                        'reason': f'Current reorder point not aligned with {avg_daily_sales:.1f} daily sales velocity'
                    })
                    
                    if opportunity['priority'] == 'LOW':
                        opportunity['priority'] = 'MEDIUM'
            
            # Fast-moving product recommendations
            if turnover_rate > 10 and days_of_supply < 14:  # High turnover, low supply
                if max_stock_level > 0:
                    suggested_max = int(max_stock_level * 1.3)  # Increase by 30%
                    opportunity['recommendations'].append({
                        'type': 'INCREASE_MAX_STOCK',
                        'description': f'Increase maximum stock level from {max_stock_level} to {suggested_max} for fast-moving product',
                        'current_value': max_stock_level,
                        'suggested_value': suggested_max,
                        'reason': f'High turnover rate ({turnover_rate:.1f}) with only {days_of_supply:.1f} days supply'
                    })
                    
                    # Estimate potential additional sales
                    additional_sales = (suggested_max - max_stock_level) * unit_price * (turnover_rate / 30)
                    opportunity['potential_impact'] = max(opportunity['potential_impact'], additional_sales)
                    
                    if opportunity['priority'] == 'LOW':
                        opportunity['priority'] = 'MEDIUM'
            
            # Add to opportunities if there are recommendations
            if opportunity['recommendations']:
                optimization_opportunities.append(opportunity)
        
        # Generate summary recommendations
        high_priority = [opp for opp in optimization_opportunities if opp['priority'] == 'HIGH']
        medium_priority = [opp for opp in optimization_opportunities if opp['priority'] == 'MEDIUM']
        
        if high_priority:
            recommendations.append(f"URGENT: {len(high_priority)} products require immediate attention to prevent stockouts")
        
        if medium_priority:
            recommendations.append(f"OPTIMIZE: {len(medium_priority)} products have inventory optimization opportunities")
        
        # Category-level recommendations
        category_analysis = {}
        for opp in optimization_opportunities:
            cat = opp['category']
            if cat not in category_analysis:
                category_analysis[cat] = {'products': 0, 'total_impact': 0, 'high_priority': 0}
            
            category_analysis[cat]['products'] += 1
            category_analysis[cat]['total_impact'] += opp['potential_impact']
            if opp['priority'] == 'HIGH':
                category_analysis[cat]['high_priority'] += 1
        
        # Sort categories by impact
        top_categories = sorted(
            category_analysis.items(),
            key=lambda x: x[1]['total_impact'],
            reverse=True
        )[:3]
        
        if top_categories:
            top_cat_names = [cat[0] for cat in top_categories]
            recommendations.append(f"Focus optimization efforts on categories: {', '.join(top_cat_names)}")
        
        # Calculate total potential impact
        total_potential_impact = sum(opp['potential_impact'] for opp in optimization_opportunities)
        
        conn.close()
        
        return {
            "success": True,
            "message": f"Generated {len(optimization_opportunities)} inventory optimization recommendations",
            "data": {
                "optimization_opportunities": optimization_opportunities,
                "summary": {
                    "total_opportunities": len(optimization_opportunities),
                    "high_priority": len(high_priority),
                    "medium_priority": len(medium_priority),
                    "total_potential_impact": total_potential_impact
                },
                "category_analysis": {
                    cat: {
                        "products": data['products'],
                        "total_impact": data['total_impact'],
                        "high_priority_count": data['high_priority']
                    }
                    for cat, data in category_analysis.items()
                },
                "recommendations": recommendations,
                "analysis_period_days": 30,
                "analyzed_at": datetime.now().isoformat()
            }
        }
        
    except sqlite3.Error as e:
        logging.error(f"Database error in generate_inventory_optimization_recommendations: {e}")
        return {
            "success": False,
            "message": f"Database error generating optimization recommendations: {e}"
        }
    except Exception as e:
        logging.error(f"Error in generate_inventory_optimization_recommendations: {e}")
        return {
            "success": False,
            "message": f"Error generating optimization recommendations: {e}"
        }


def generate_forecasting_context(product_id: Optional[str] = None, store_id: Optional[str] = None, 
                               days_back: int = FORECASTING_CONTEXT_DAYS) -> Dict[str, Any]:
    """
    Generate comprehensive inventory context for demand forecasting models.
    
    Provides historical inventory patterns, seasonality, and trends for forecasting context.
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Base query for inventory context
        base_query = """
            SELECT 
                il.product_id,
                il.store_id,
                il.current_stock,
                il.available_stock,
                il.reorder_point,
                il.max_stock_level,
                il.last_updated,
                p.name as product_name,
                p.category,
                p.subcategory,
                p.unit_price,
                s.name as store_name,
                s.location,
                s.region,
                -- Sales velocity and patterns
                COALESCE(sales_data.total_sales, 0) as total_sales,
                COALESCE(sales_data.avg_daily_sales, 0) as avg_daily_sales,
                COALESCE(sales_data.sales_trend, 0) as sales_trend,
                COALESCE(sales_data.sales_volatility, 0) as sales_volatility,
                -- Stockout history
                COALESCE(stockout_data.stockout_frequency, 0) as stockout_frequency,
                COALESCE(stockout_data.avg_stockout_duration, 0) as avg_stockout_duration,
                COALESCE(stockout_data.total_lost_sales, 0) as total_lost_sales
            FROM inventory_levels il
            JOIN products p ON il.product_id = p.id
            JOIN stores s ON il.store_id = s.id
            LEFT JOIN (
                SELECT 
                    st.product_id,
                    st.store_id,
                    SUM(st.quantity) as total_sales,
                    AVG(daily_sales.daily_quantity) as avg_daily_sales,
                    -- Calculate trend using linear regression approximation
                    (COUNT(*) * SUM(daily_sales.day_number * daily_sales.daily_quantity) - 
                     SUM(daily_sales.day_number) * SUM(daily_sales.daily_quantity)) /
                    (COUNT(*) * SUM(daily_sales.day_number * daily_sales.day_number) - 
                     SUM(daily_sales.day_number) * SUM(daily_sales.day_number)) as sales_trend,
                    -- Calculate volatility as coefficient of variation
                    CASE 
                        WHEN AVG(daily_sales.daily_quantity) > 0 
                        THEN (SQRT(AVG(daily_sales.daily_quantity * daily_sales.daily_quantity) - 
                                  AVG(daily_sales.daily_quantity) * AVG(daily_sales.daily_quantity)) / 
                              AVG(daily_sales.daily_quantity))
                        ELSE 0 
                    END as sales_volatility
                FROM sales_transactions st
                JOIN (
                    SELECT 
                        product_id,
                        store_id,
                        DATE(transaction_date) as sale_date,
                        SUM(quantity) as daily_quantity,
                        ROW_NUMBER() OVER (PARTITION BY product_id, store_id ORDER BY DATE(transaction_date)) as day_number
                    FROM sales_transactions
                    WHERE transaction_date >= ?
                    GROUP BY product_id, store_id, DATE(transaction_date)
                ) daily_sales ON st.product_id = daily_sales.product_id 
                    AND st.store_id = daily_sales.store_id
                    AND DATE(st.transaction_date) = daily_sales.sale_date
                WHERE st.transaction_date >= ?
                GROUP BY st.product_id, st.store_id
            ) sales_data ON il.product_id = sales_data.product_id AND il.store_id = sales_data.store_id
            LEFT JOIN (
                SELECT 
                    so.product_id,
                    so.store_id,
                    COUNT(*) as stockout_frequency,
                    AVG(so.duration_hours) as avg_stockout_duration,
                    SUM(so.lost_sales_estimate) as total_lost_sales
                FROM stockout_events so
                WHERE so.stockout_date >= ?
                GROUP BY so.product_id, so.store_id
            ) stockout_data ON il.product_id = stockout_data.product_id AND il.store_id = stockout_data.store_id
            WHERE p.active = 1 AND s.active = 1
        """
        
        cutoff_date = datetime.now() - timedelta(days=days_back)
        params = [cutoff_date, cutoff_date, cutoff_date]
        
        if product_id:
            base_query += " AND il.product_id = ?"
            params.append(product_id)
        
        if store_id:
            base_query += " AND il.store_id = ?"
            params.append(store_id)
        
        base_query += " ORDER BY p.category, il.product_id, s.location"
        
        cursor.execute(base_query, params)
        rows = cursor.fetchall()
        
        if not rows:
            return {
                "success": True,
                "message": "No inventory context data available",
                "data": {
                    "forecasting_context": [],
                    "summary": {
                        "total_products": 0,
                        "total_locations": 0,
                        "avg_inventory_level": 0,
                        "avg_sales_velocity": 0
                    }
                }
            }
        
        # Process forecasting context data
        forecasting_context = []
        total_inventory_value = 0
        total_sales_velocity = 0
        categories = set()
        locations = set()
        
        for row in rows:
            # Calculate key forecasting metrics
            current_stock = row['current_stock']
            available_stock = row['available_stock']
            avg_daily_sales = row['avg_daily_sales'] or 0
            sales_trend = row['sales_trend'] or 0
            sales_volatility = row['sales_volatility'] or 0
            unit_price = row['unit_price'] or 0
            
            # Calculate days of supply
            days_of_supply = available_stock / avg_daily_sales if avg_daily_sales > 0 else float('inf')
            
            # Calculate inventory turnover rate
            turnover_rate = (row['total_sales'] or 0) / current_stock if current_stock > 0 else 0
            
            # Calculate stockout risk score (0-100)
            stockout_risk_score = 0
            if row['stockout_frequency'] > 0:
                stockout_risk_score += min(row['stockout_frequency'] * 10, 50)
            if days_of_supply < 7:
                stockout_risk_score += 30
            if available_stock <= (row['reorder_point'] or 0):
                stockout_risk_score += 20
            stockout_risk_score = min(stockout_risk_score, 100)
            
            # Calculate demand variability score
            demand_variability = "Low"
            if sales_volatility > 1.0:
                demand_variability = "High"
            elif sales_volatility > 0.5:
                demand_variability = "Medium"
            
            # Determine seasonal pattern (simplified)
            seasonal_pattern = "Stable"
            if abs(sales_trend) > 0.1:
                seasonal_pattern = "Growing" if sales_trend > 0 else "Declining"
            
            context_item = {
                'product_id': row['product_id'],
                'store_id': row['store_id'],
                'product_name': row['product_name'],
                'category': row['category'],
                'subcategory': row['subcategory'],
                'store_name': row['store_name'],
                'location': row['location'],
                'region': row['region'],
                'inventory_metrics': {
                    'current_stock': current_stock,
                    'available_stock': available_stock,
                    'reorder_point': row['reorder_point'],
                    'max_stock_level': row['max_stock_level'],
                    'days_of_supply': min(days_of_supply, 999),
                    'turnover_rate': turnover_rate,
                    'inventory_value': current_stock * unit_price
                },
                'sales_patterns': {
                    'total_sales': row['total_sales'] or 0,
                    'avg_daily_sales': avg_daily_sales,
                    'sales_trend': sales_trend,
                    'sales_volatility': sales_volatility,
                    'demand_variability': demand_variability,
                    'seasonal_pattern': seasonal_pattern
                },
                'risk_indicators': {
                    'stockout_frequency': row['stockout_frequency'] or 0,
                    'avg_stockout_duration': row['avg_stockout_duration'] or 0,
                    'total_lost_sales': row['total_lost_sales'] or 0,
                    'stockout_risk_score': stockout_risk_score
                },
                'forecasting_features': {
                    'lead_time_demand': avg_daily_sales * 7,  # Assume 7-day lead time
                    'safety_stock_needed': max(0, (row['reorder_point'] or 0) - (avg_daily_sales * 7)),
                    'optimal_order_quantity': calculate_eoq(avg_daily_sales, unit_price),
                    'forecast_horizon_days': 30,
                    'confidence_level': max(0.5, 1.0 - (sales_volatility / 2))
                },
                'last_updated': row['last_updated']
            }
            
            forecasting_context.append(context_item)
            
            # Aggregate statistics
            total_inventory_value += context_item['inventory_metrics']['inventory_value']
            total_sales_velocity += avg_daily_sales
            categories.add(row['category'])
            locations.add(row['location'])
        
        # Calculate summary statistics
        total_products = len(forecasting_context)
        avg_inventory_level = sum(item['inventory_metrics']['current_stock'] for item in forecasting_context) / total_products if total_products > 0 else 0
        avg_sales_velocity = total_sales_velocity / total_products if total_products > 0 else 0
        
        # Generate forecasting insights
        insights = []
        recommendations = []
        
        high_risk_products = [item for item in forecasting_context if item['risk_indicators']['stockout_risk_score'] > 70]
        if high_risk_products:
            insights.append(f"{len(high_risk_products)} products have high stockout risk")
            recommendations.append("Prioritize safety stock optimization for high-risk products")
        
        volatile_demand_products = [item for item in forecasting_context if item['sales_patterns']['demand_variability'] == 'High']
        if volatile_demand_products:
            insights.append(f"{len(volatile_demand_products)} products have high demand variability")
            recommendations.append("Use ensemble forecasting models for volatile demand products")
        
        growing_products = [item for item in forecasting_context if item['sales_patterns']['seasonal_pattern'] == 'Growing']
        if growing_products:
            insights.append(f"{len(growing_products)} products show growing demand trends")
            recommendations.append("Increase inventory levels for growing demand products")
        
        conn.close()
        
        return {
            "success": True,
            "message": f"Generated forecasting context for {total_products} products",
            "data": {
                "forecasting_context": forecasting_context,
                "summary": {
                    "total_products": total_products,
                    "total_locations": len(locations),
                    "total_categories": len(categories),
                    "avg_inventory_level": round(avg_inventory_level, 2),
                    "avg_sales_velocity": round(avg_sales_velocity, 2),
                    "total_inventory_value": round(total_inventory_value, 2),
                    "high_risk_products": len(high_risk_products),
                    "volatile_demand_products": len(volatile_demand_products)
                },
                "insights": insights,
                "recommendations": recommendations,
                "analysis_period_days": days_back,
                "generated_at": datetime.now().isoformat()
            }
        }
        
    except sqlite3.Error as e:
        logging.error(f"Database error in generate_forecasting_context: {e}")
        return {
            "success": False,
            "message": f"Database error generating forecasting context: {e}"
        }
    except Exception as e:
        logging.error(f"Error in generate_forecasting_context: {e}")
        return {
            "success": False,
            "message": f"Error generating forecasting context: {e}"
        }


def calculate_eoq(daily_demand: float, unit_cost: float, holding_cost_rate: float = 0.25, 
                  ordering_cost: float = 50.0) -> float:
    """
    Calculate Economic Order Quantity (EOQ) for optimal inventory ordering.
    
    Args:
        daily_demand: Average daily demand
        unit_cost: Cost per unit
        holding_cost_rate: Annual holding cost as percentage of unit cost
        ordering_cost: Fixed cost per order
    
    Returns:
        Optimal order quantity
    """
    try:
        annual_demand = daily_demand * 365
        annual_holding_cost = unit_cost * holding_cost_rate
        
        if annual_holding_cost <= 0:
            return daily_demand * 30  # Default to 30-day supply
        
        eoq = (2 * annual_demand * ordering_cost / annual_holding_cost) ** 0.5
        return max(eoq, daily_demand)  # At least one day's demand
        
    except (ValueError, ZeroDivisionError):
        return daily_demand * 30  # Default fallback


def integrate_sales_inventory_data(days_back: int = 30) -> Dict[str, Any]:
    """
    Create integrated analysis combining sales and inventory data for comprehensive insights.
    
    Provides unified view of sales performance and inventory efficiency.
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        # Comprehensive integration query
        cursor.execute("""
            SELECT 
                p.id as product_id,
                p.name as product_name,
                p.category,
                p.subcategory,
                p.unit_price,
                s.id as store_id,
                s.name as store_name,
                s.location,
                s.region,
                -- Current inventory state
                il.current_stock,
                il.available_stock,
                il.reorder_point,
                il.max_stock_level,
                -- Sales performance
                COALESCE(sales_summary.total_quantity, 0) as total_sales_quantity,
                COALESCE(sales_summary.total_revenue, 0) as total_sales_revenue,
                COALESCE(sales_summary.avg_daily_quantity, 0) as avg_daily_sales,
                COALESCE(sales_summary.sales_days, 0) as active_sales_days,
                COALESCE(sales_summary.avg_transaction_size, 0) as avg_transaction_size,
                -- Inventory efficiency metrics
                CASE 
                    WHEN il.current_stock > 0 AND sales_summary.total_quantity > 0
                    THEN sales_summary.total_quantity / il.current_stock
                    ELSE 0 
                END as inventory_turnover,
                CASE 
                    WHEN sales_summary.avg_daily_quantity > 0
                    THEN il.available_stock / sales_summary.avg_daily_quantity
                    ELSE 999 
                END as days_of_supply,
                -- Stockout impact
                COALESCE(stockout_summary.stockout_events, 0) as stockout_events,
                COALESCE(stockout_summary.total_stockout_hours, 0) as total_stockout_hours,
                COALESCE(stockout_summary.lost_sales_amount, 0) as lost_sales_amount
            FROM products p
            JOIN inventory_levels il ON p.id = il.product_id
            JOIN stores s ON il.store_id = s.id
            LEFT JOIN (
                SELECT 
                    st.product_id,
                    st.store_id,
                    SUM(st.quantity) as total_quantity,
                    SUM(st.total_amount) as total_revenue,
                    AVG(daily_sales.daily_quantity) as avg_daily_quantity,
                    COUNT(DISTINCT DATE(st.transaction_date)) as sales_days,
                    AVG(st.quantity) as avg_transaction_size
                FROM sales_transactions st
                JOIN (
                    SELECT 
                        product_id,
                        store_id,
                        DATE(transaction_date) as sale_date,
                        SUM(quantity) as daily_quantity
                    FROM sales_transactions
                    WHERE transaction_date >= ?
                    GROUP BY product_id, store_id, DATE(transaction_date)
                ) daily_sales ON st.product_id = daily_sales.product_id 
                    AND st.store_id = daily_sales.store_id
                    AND DATE(st.transaction_date) = daily_sales.sale_date
                WHERE st.transaction_date >= ?
                GROUP BY st.product_id, st.store_id
            ) sales_summary ON p.id = sales_summary.product_id AND il.store_id = sales_summary.store_id
            LEFT JOIN (
                SELECT 
                    so.product_id,
                    so.store_id,
                    COUNT(*) as stockout_events,
                    SUM(so.duration_hours) as total_stockout_hours,
                    SUM(so.lost_sales_estimate) as lost_sales_amount
                FROM stockout_events so
                WHERE so.stockout_date >= ?
                GROUP BY so.product_id, so.store_id
            ) stockout_summary ON p.id = stockout_summary.product_id AND il.store_id = stockout_summary.store_id
            WHERE p.active = 1 AND s.active = 1
            ORDER BY p.category, total_sales_revenue DESC
        """, (cutoff_date, cutoff_date, cutoff_date))
        
        rows = cursor.fetchall()
        
        if not rows:
            return {
                "success": True,
                "message": "No integrated sales-inventory data available",
                "data": {
                    "integrated_analysis": [],
                    "summary": {}
                }
            }
        
        # Process integrated data
        integrated_analysis = []
        category_performance = {}
        location_performance = {}
        
        total_revenue = 0
        total_inventory_value = 0
        total_lost_sales = 0
        
        for row in rows:
            product_id = row['product_id']
            category = row['category']
            location = row['location']
            
            # Calculate performance metrics
            sales_quantity = row['total_sales_quantity'] or 0
            sales_revenue = row['total_sales_revenue'] or 0
            current_stock = row['current_stock'] or 0
            available_stock = row['available_stock'] or 0
            unit_price = row['unit_price'] or 0
            avg_daily_sales = row['avg_daily_sales'] or 0
            inventory_turnover = row['inventory_turnover'] or 0
            days_of_supply = min(row['days_of_supply'] or 999, 999)
            lost_sales = row['lost_sales_amount'] or 0
            
            # Calculate efficiency scores
            inventory_efficiency_score = 0
            if inventory_turnover > 0:
                if inventory_turnover > 12:  # High turnover
                    inventory_efficiency_score = 90
                elif inventory_turnover > 6:  # Medium turnover
                    inventory_efficiency_score = 70
                elif inventory_turnover > 2:  # Low turnover
                    inventory_efficiency_score = 50
                else:  # Very low turnover
                    inventory_efficiency_score = 30
            
            # Adjust for stockouts (penalty)
            if row['stockout_events'] > 0:
                inventory_efficiency_score = max(0, inventory_efficiency_score - (row['stockout_events'] * 10))
            
            # Calculate sales performance score
            sales_performance_score = 50  # Base score
            if sales_revenue > 0:
                # Adjust based on sales consistency
                if row['active_sales_days'] > (days_back * 0.8):  # Consistent sales
                    sales_performance_score += 30
                elif row['active_sales_days'] > (days_back * 0.5):  # Moderate sales
                    sales_performance_score += 15
                
                # Adjust based on transaction size
                if row['avg_transaction_size'] > 2:  # Bulk purchases
                    sales_performance_score += 10
            
            sales_performance_score = min(sales_performance_score, 100)
            
            # Overall performance score
            overall_performance_score = (inventory_efficiency_score + sales_performance_score) / 2
            
            # Performance classification
            if overall_performance_score >= 80:
                performance_class = "Excellent"
            elif overall_performance_score >= 65:
                performance_class = "Good"
            elif overall_performance_score >= 50:
                performance_class = "Average"
            elif overall_performance_score >= 35:
                performance_class = "Poor"
            else:
                performance_class = "Critical"
            
            analysis_item = {
                'product_id': product_id,
                'product_name': row['product_name'],
                'category': category,
                'subcategory': row['subcategory'],
                'store_id': row['store_id'],
                'store_name': row['store_name'],
                'location': location,
                'region': row['region'],
                'inventory_data': {
                    'current_stock': current_stock,
                    'available_stock': available_stock,
                    'reorder_point': row['reorder_point'],
                    'max_stock_level': row['max_stock_level'],
                    'inventory_value': current_stock * unit_price,
                    'days_of_supply': days_of_supply
                },
                'sales_data': {
                    'total_quantity': sales_quantity,
                    'total_revenue': sales_revenue,
                    'avg_daily_sales': avg_daily_sales,
                    'active_sales_days': row['active_sales_days'],
                    'avg_transaction_size': row['avg_transaction_size']
                },
                'performance_metrics': {
                    'inventory_turnover': inventory_turnover,
                    'inventory_efficiency_score': round(inventory_efficiency_score, 1),
                    'sales_performance_score': round(sales_performance_score, 1),
                    'overall_performance_score': round(overall_performance_score, 1),
                    'performance_class': performance_class
                },
                'risk_factors': {
                    'stockout_events': row['stockout_events'],
                    'total_stockout_hours': row['total_stockout_hours'],
                    'lost_sales_amount': lost_sales,
                    'overstock_risk': 'High' if days_of_supply > 60 else 'Medium' if days_of_supply > 30 else 'Low',
                    'understock_risk': 'High' if available_stock <= (row['reorder_point'] or 0) else 'Low'
                }
            }
            
            integrated_analysis.append(analysis_item)
            
            # Aggregate by category
            if category not in category_performance:
                category_performance[category] = {
                    'products': 0, 'total_revenue': 0, 'total_inventory_value': 0,
                    'avg_turnover': 0, 'stockout_events': 0, 'lost_sales': 0
                }
            
            cat_perf = category_performance[category]
            cat_perf['products'] += 1
            cat_perf['total_revenue'] += sales_revenue
            cat_perf['total_inventory_value'] += current_stock * unit_price
            cat_perf['avg_turnover'] += inventory_turnover
            cat_perf['stockout_events'] += row['stockout_events'] or 0
            cat_perf['lost_sales'] += lost_sales
            
            # Aggregate by location
            if location not in location_performance:
                location_performance[location] = {
                    'products': 0, 'total_revenue': 0, 'total_inventory_value': 0,
                    'avg_turnover': 0, 'stockout_events': 0, 'lost_sales': 0
                }
            
            loc_perf = location_performance[location]
            loc_perf['products'] += 1
            loc_perf['total_revenue'] += sales_revenue
            loc_perf['total_inventory_value'] += current_stock * unit_price
            loc_perf['avg_turnover'] += inventory_turnover
            loc_perf['stockout_events'] += row['stockout_events'] or 0
            loc_perf['lost_sales'] += lost_sales
            
            # Overall totals
            total_revenue += sales_revenue
            total_inventory_value += current_stock * unit_price
            total_lost_sales += lost_sales
        
        # Finalize category and location averages
        for cat_data in category_performance.values():
            if cat_data['products'] > 0:
                cat_data['avg_turnover'] = cat_data['avg_turnover'] / cat_data['products']
        
        for loc_data in location_performance.values():
            if loc_data['products'] > 0:
                loc_data['avg_turnover'] = loc_data['avg_turnover'] / loc_data['products']
        
        # Generate insights and recommendations
        insights = []
        recommendations = []
        
        # Performance insights
        excellent_performers = [item for item in integrated_analysis if item['performance_metrics']['performance_class'] == 'Excellent']
        poor_performers = [item for item in integrated_analysis if item['performance_metrics']['performance_class'] in ['Poor', 'Critical']]
        
        if excellent_performers:
            insights.append(f"{len(excellent_performers)} products are excellent performers")
            recommendations.append("Replicate success patterns from excellent performers across similar products")
        
        if poor_performers:
            insights.append(f"{len(poor_performers)} products need immediate attention")
            recommendations.append("Focus optimization efforts on poor-performing products")
        
        # Inventory insights
        overstock_items = [item for item in integrated_analysis if item['risk_factors']['overstock_risk'] == 'High']
        if overstock_items:
            insights.append(f"{len(overstock_items)} products have overstock risk")
            recommendations.append("Reduce inventory levels for overstocked items")
        
        # Revenue insights
        if total_lost_sales > (total_revenue * 0.05):  # More than 5% revenue loss
            insights.append(f"Significant revenue loss from stockouts: ${total_lost_sales:,.2f}")
            recommendations.append("Implement proactive stockout prevention measures")
        
        conn.close()
        
        return {
            "success": True,
            "message": f"Integrated analysis completed for {len(integrated_analysis)} products",
            "data": {
                "integrated_analysis": integrated_analysis,
                "summary": {
                    "total_products": len(integrated_analysis),
                    "total_revenue": round(total_revenue, 2),
                    "total_inventory_value": round(total_inventory_value, 2),
                    "total_lost_sales": round(total_lost_sales, 2),
                    "revenue_loss_percentage": round((total_lost_sales / total_revenue * 100) if total_revenue > 0 else 0, 2),
                    "excellent_performers": len(excellent_performers),
                    "poor_performers": len(poor_performers)
                },
                "category_performance": {
                    cat: {
                        "products": data['products'],
                        "total_revenue": round(data['total_revenue'], 2),
                        "total_inventory_value": round(data['total_inventory_value'], 2),
                        "avg_turnover": round(data['avg_turnover'], 2),
                        "stockout_events": data['stockout_events'],
                        "lost_sales": round(data['lost_sales'], 2)
                    }
                    for cat, data in category_performance.items()
                },
                "location_performance": {
                    loc: {
                        "products": data['products'],
                        "total_revenue": round(data['total_revenue'], 2),
                        "total_inventory_value": round(data['total_inventory_value'], 2),
                        "avg_turnover": round(data['avg_turnover'], 2),
                        "stockout_events": data['stockout_events'],
                        "lost_sales": round(data['lost_sales'], 2)
                    }
                    for loc, data in location_performance.items()
                },
                "insights": insights,
                "recommendations": recommendations,
                "analysis_period_days": days_back,
                "analyzed_at": datetime.now().isoformat()
            }
        }
        
    except sqlite3.Error as e:
        logging.error(f"Database error in integrate_sales_inventory_data: {e}")
        return {
            "success": False,
            "message": f"Database error integrating sales-inventory data: {e}"
        }
    except Exception as e:
        logging.error(f"Error in integrate_sales_inventory_data: {e}")
        return {
            "success": False,
            "message": f"Error integrating sales-inventory data: {e}"
        }


def analyze_lead_times_and_optimize_reorder_points(days_back: int = LEAD_TIME_ANALYSIS_DAYS) -> Dict[str, Any]:
    """
    Analyze supplier lead times and optimize reorder points for improved inventory management.
    
    Provides lead time analysis and reorder point optimization recommendations.
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        # Analyze lead times using stockout and restock patterns
        cursor.execute("""
            SELECT 
                p.id as product_id,
                p.name as product_name,
                p.category,
                p.unit_price,
                s.id as store_id,
                s.name as store_name,
                s.location,
                il.current_stock,
                il.available_stock,
                il.reorder_point,
                il.max_stock_level,
                -- Sales velocity for lead time calculation
                COALESCE(sales_velocity.avg_daily_sales, 0) as avg_daily_sales,
                COALESCE(sales_velocity.sales_std_dev, 0) as sales_std_dev,
                -- Lead time estimation from stockout events
                COALESCE(lead_time_data.avg_lead_time_hours, 168) as avg_lead_time_hours,  -- Default 7 days
                COALESCE(lead_time_data.lead_time_variability, 24) as lead_time_variability,
                COALESCE(lead_time_data.stockout_frequency, 0) as stockout_frequency,
                -- Service level indicators
                COALESCE(service_level.fill_rate, 1.0) as fill_rate,
                COALESCE(service_level.stockout_days, 0) as stockout_days
            FROM products p
            JOIN inventory_levels il ON p.id = il.product_id
            JOIN stores s ON il.store_id = s.id
            LEFT JOIN (
                SELECT 
                    st.product_id,
                    st.store_id,
                    AVG(daily_sales.daily_quantity) as avg_daily_sales,
                    SQRT(AVG(daily_sales.daily_quantity * daily_sales.daily_quantity) - 
                         AVG(daily_sales.daily_quantity) * AVG(daily_sales.daily_quantity)) as sales_std_dev
                FROM sales_transactions st
                JOIN (
                    SELECT 
                        product_id,
                        store_id,
                        DATE(transaction_date) as sale_date,
                        SUM(quantity) as daily_quantity
                    FROM sales_transactions
                    WHERE transaction_date >= ?
                    GROUP BY product_id, store_id, DATE(transaction_date)
                ) daily_sales ON st.product_id = daily_sales.product_id 
                    AND st.store_id = daily_sales.store_id
                    AND DATE(st.transaction_date) = daily_sales.sale_date
                WHERE st.transaction_date >= ?
                GROUP BY st.product_id, st.store_id
            ) sales_velocity ON p.id = sales_velocity.product_id AND il.store_id = sales_velocity.store_id
            LEFT JOIN (
                SELECT 
                    so.product_id,
                    so.store_id,
                    AVG(so.duration_hours) as avg_lead_time_hours,
                    SQRT(AVG(so.duration_hours * so.duration_hours) - 
                         AVG(so.duration_hours) * AVG(so.duration_hours)) as lead_time_variability,
                    COUNT(*) as stockout_frequency
                FROM stockout_events so
                WHERE so.stockout_date >= ? AND so.restock_date IS NOT NULL
                GROUP BY so.product_id, so.store_id
            ) lead_time_data ON p.id = lead_time_data.product_id AND il.store_id = lead_time_data.store_id
            LEFT JOIN (
                SELECT 
                    st.product_id,
                    st.store_id,
                    -- Calculate fill rate as percentage of demand met
                    CASE 
                        WHEN (SUM(st.quantity) + COALESCE(lost_sales.total_lost_quantity, 0)) > 0
                        THEN SUM(st.quantity) / (SUM(st.quantity) + COALESCE(lost_sales.total_lost_quantity, 0))
                        ELSE 1.0 
                    END as fill_rate,
                    COALESCE(stockout_days.days_out_of_stock, 0) as stockout_days
                FROM sales_transactions st
                LEFT JOIN (
                    SELECT 
                        so.product_id,
                        so.store_id,
                        SUM(so.lost_sales_estimate / COALESCE(p.unit_price, 1)) as total_lost_quantity
                    FROM stockout_events so
                    JOIN products p ON so.product_id = p.id
                    WHERE so.stockout_date >= ?
                    GROUP BY so.product_id, so.store_id
                ) lost_sales ON st.product_id = lost_sales.product_id AND st.store_id = lost_sales.store_id
                LEFT JOIN (
                    SELECT 
                        so.product_id,
                        so.store_id,
                        SUM(COALESCE(so.duration_hours, 0) / 24.0) as days_out_of_stock
                    FROM stockout_events so
                    WHERE so.stockout_date >= ?
                    GROUP BY so.product_id, so.store_id
                ) stockout_days ON st.product_id = stockout_days.product_id AND st.store_id = stockout_days.store_id
                WHERE st.transaction_date >= ?
                GROUP BY st.product_id, st.store_id
            ) service_level ON p.id = service_level.product_id AND il.store_id = service_level.store_id
            WHERE p.active = 1 AND s.active = 1
            ORDER BY p.category, stockout_frequency DESC
        """, (cutoff_date, cutoff_date, cutoff_date, cutoff_date, cutoff_date, cutoff_date))
        
        rows = cursor.fetchall()
        
        if not rows:
            return {
                "success": True,
                "message": "No lead time analysis data available",
                "data": {
                    "lead_time_analysis": [],
                    "optimization_recommendations": []
                }
            }
        
        # Process lead time analysis
        lead_time_analysis = []
        optimization_recommendations = []
        
        for row in rows:
            product_id = row['product_id']
            avg_daily_sales = row['avg_daily_sales'] or 0
            sales_std_dev = row['sales_std_dev'] or 0
            avg_lead_time_hours = row['avg_lead_time_hours'] or 168  # Default 7 days
            lead_time_variability = row['lead_time_variability'] or 24
            current_reorder_point = row['reorder_point'] or 0
            fill_rate = row['fill_rate'] or 1.0
            stockout_frequency = row['stockout_frequency'] or 0
            
            # Convert lead time to days
            avg_lead_time_days = avg_lead_time_hours / 24
            lead_time_std_dev_days = lead_time_variability / 24
            
            # Calculate optimal reorder point using safety stock formula
            # ROP = (Average Lead Time × Average Demand) + Safety Stock
            # Safety Stock = Z-score × √(Lead Time × Demand Variance + Demand² × Lead Time Variance)
            
            # Target service level (95% = Z-score of 1.645, 99% = 2.33)
            target_service_level = 0.95
            z_score = 1.645  # For 95% service level
            
            # If current fill rate is low, increase target service level
            if fill_rate < 0.90:
                target_service_level = 0.99
                z_score = 2.33
            
            # Calculate safety stock
            demand_variance = sales_std_dev ** 2
            lead_time_variance = lead_time_std_dev_days ** 2
            
            safety_stock_variance = (avg_lead_time_days * demand_variance + 
                                   (avg_daily_sales ** 2) * lead_time_variance)
            safety_stock = z_score * (safety_stock_variance ** 0.5) if safety_stock_variance > 0 else avg_daily_sales
            
            # Calculate optimal reorder point
            optimal_reorder_point = int((avg_lead_time_days * avg_daily_sales) + safety_stock)
            
            # Ensure minimum reorder point
            optimal_reorder_point = max(optimal_reorder_point, int(avg_daily_sales * 3))  # At least 3 days
            
            # Calculate current vs optimal comparison
            reorder_point_difference = optimal_reorder_point - current_reorder_point
            improvement_needed = abs(reorder_point_difference) > (avg_daily_sales * 2)  # Significant if >2 days demand
            
            # Calculate expected service level with current reorder point
            if avg_daily_sales > 0 and safety_stock > 0:
                current_safety_stock = max(0, current_reorder_point - (avg_lead_time_days * avg_daily_sales))
                current_z_score = current_safety_stock / (safety_stock_variance ** 0.5) if safety_stock_variance > 0 else 0
                # Approximate service level from z-score (simplified)
                if current_z_score >= 2.33:
                    expected_service_level = 0.99
                elif current_z_score >= 1.645:
                    expected_service_level = 0.95
                elif current_z_score >= 1.28:
                    expected_service_level = 0.90
                elif current_z_score >= 0.84:
                    expected_service_level = 0.80
                else:
                    expected_service_level = 0.70
            else:
                expected_service_level = fill_rate
            
            # Lead time classification
            if avg_lead_time_days <= 3:
                lead_time_class = "Short"
            elif avg_lead_time_days <= 7:
                lead_time_class = "Medium"
            elif avg_lead_time_days <= 14:
                lead_time_class = "Long"
            else:
                lead_time_class = "Very Long"
            
            # Demand variability classification
            if avg_daily_sales > 0:
                cv = sales_std_dev / avg_daily_sales  # Coefficient of variation
                if cv <= 0.25:
                    demand_variability = "Low"
                elif cv <= 0.75:
                    demand_variability = "Medium"
                else:
                    demand_variability = "High"
            else:
                demand_variability = "Unknown"
            
            analysis_item = {
                'product_id': product_id,
                'product_name': row['product_name'],
                'category': row['category'],
                'store_id': row['store_id'],
                'store_name': row['store_name'],
                'location': row['location'],
                'current_inventory': {
                    'current_stock': row['current_stock'],
                    'available_stock': row['available_stock'],
                    'current_reorder_point': current_reorder_point,
                    'max_stock_level': row['max_stock_level']
                },
                'demand_patterns': {
                    'avg_daily_sales': round(avg_daily_sales, 2),
                    'sales_std_dev': round(sales_std_dev, 2),
                    'demand_variability': demand_variability,
                    'coefficient_of_variation': round(sales_std_dev / avg_daily_sales, 2) if avg_daily_sales > 0 else 0
                },
                'lead_time_analysis': {
                    'avg_lead_time_days': round(avg_lead_time_days, 1),
                    'lead_time_std_dev_days': round(lead_time_std_dev_days, 1),
                    'lead_time_class': lead_time_class,
                    'stockout_frequency': stockout_frequency
                },
                'service_level_metrics': {
                    'current_fill_rate': round(fill_rate, 3),
                    'expected_service_level': round(expected_service_level, 3),
                    'target_service_level': target_service_level,
                    'stockout_days': row['stockout_days']
                },
                'optimization_results': {
                    'optimal_reorder_point': optimal_reorder_point,
                    'current_reorder_point': current_reorder_point,
                    'reorder_point_difference': reorder_point_difference,
                    'safety_stock_needed': round(safety_stock, 1),
                    'improvement_needed': improvement_needed,
                    'expected_inventory_reduction': max(0, current_reorder_point - optimal_reorder_point) if reorder_point_difference < 0 else 0,
                    'expected_service_improvement': max(0, target_service_level - expected_service_level)
                }
            }
            
            lead_time_analysis.append(analysis_item)
            
            # Generate specific recommendations
            if improvement_needed:
                if reorder_point_difference > 0:
                    # Need to increase reorder point
                    recommendation = {
                        'product_id': product_id,
                        'product_name': row['product_name'],
                        'store_name': row['store_name'],
                        'recommendation_type': 'INCREASE_REORDER_POINT',
                        'current_value': current_reorder_point,
                        'recommended_value': optimal_reorder_point,
                        'reason': f'Current service level ({expected_service_level:.1%}) below target ({target_service_level:.1%})',
                        'expected_benefit': f'Improve service level to {target_service_level:.1%}',
                        'priority': 'HIGH' if fill_rate < 0.90 else 'MEDIUM'
                    }
                else:
                    # Can reduce reorder point
                    recommendation = {
                        'product_id': product_id,
                        'product_name': row['product_name'],
                        'store_name': row['store_name'],
                        'recommendation_type': 'REDUCE_REORDER_POINT',
                        'current_value': current_reorder_point,
                        'recommended_value': optimal_reorder_point,
                        'reason': f'Current reorder point is {abs(reorder_point_difference)} units too high',
                        'expected_benefit': f'Reduce inventory carrying cost while maintaining {target_service_level:.1%} service level',
                        'priority': 'MEDIUM'
                    }
                
                optimization_recommendations.append(recommendation)
        
        # Generate summary insights
        total_products = len(lead_time_analysis)
        high_priority_recommendations = [r for r in optimization_recommendations if r['priority'] == 'HIGH']
        products_needing_increase = [r for r in optimization_recommendations if r['recommendation_type'] == 'INCREASE_REORDER_POINT']
        products_allowing_reduction = [r for r in optimization_recommendations if r['recommendation_type'] == 'REDUCE_REORDER_POINT']
        
        # Calculate potential inventory impact
        total_inventory_reduction = sum(
            item['optimization_results']['expected_inventory_reduction'] 
            for item in lead_time_analysis
        )
        
        avg_service_level = sum(item['service_level_metrics']['current_fill_rate'] for item in lead_time_analysis) / total_products if total_products > 0 else 0
        
        insights = []
        recommendations_summary = []
        
        if high_priority_recommendations:
            insights.append(f"{len(high_priority_recommendations)} products need urgent reorder point adjustments")
            recommendations_summary.append("Prioritize high-priority reorder point optimizations to prevent stockouts")
        
        if products_allowing_reduction:
            insights.append(f"{len(products_allowing_reduction)} products can reduce inventory levels")
            recommendations_summary.append("Implement reorder point reductions to free up working capital")
        
        if avg_service_level < 0.95:
            insights.append(f"Average service level ({avg_service_level:.1%}) below target (95%)")
            recommendations_summary.append("Focus on improving overall service level through better inventory management")
        
        if total_inventory_reduction > 0:
            insights.append(f"Potential inventory reduction: {total_inventory_reduction:.0f} units")
            recommendations_summary.append("Optimize reorder points to reduce excess inventory")
        
        conn.close()
        
        return {
            "success": True,
            "message": f"Lead time analysis completed for {total_products} products",
            "data": {
                "lead_time_analysis": lead_time_analysis,
                "optimization_recommendations": optimization_recommendations,
                "summary": {
                    "total_products_analyzed": total_products,
                    "high_priority_recommendations": len(high_priority_recommendations),
                    "products_needing_increase": len(products_needing_increase),
                    "products_allowing_reduction": len(products_allowing_reduction),
                    "avg_current_service_level": round(avg_service_level, 3),
                    "potential_inventory_reduction": round(total_inventory_reduction, 0)
                },
                "insights": insights,
                "recommendations": recommendations_summary,
                "analysis_period_days": days_back,
                "analyzed_at": datetime.now().isoformat()
            }
        }
        
    except sqlite3.Error as e:
        logging.error(f"Database error in analyze_lead_times_and_optimize_reorder_points: {e}")
        return {
            "success": False,
            "message": f"Database error analyzing lead times: {e}"
        }
    except Exception as e:
        logging.error(f"Error in analyze_lead_times_and_optimize_reorder_points: {e}")
        return {
            "success": False,
            "message": f"Error analyzing lead times: {e}"
        }


# --- MCP Server Setup ---
server = Server("inventory-mcp-server")


@server.list_tools()
async def handle_list_tools() -> List[mcp_types.Tool]:
    """List available inventory management tools."""
    return [
        mcp_types.Tool(
            name="get_current_inventory",
            description="Get current inventory snapshot across all locations or specific location",
            inputSchema={
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "Optional location filter (store location or region)"
                    }
                }
            }
        ),
        mcp_types.Tool(
            name="analyze_stockout_patterns",
            description="Analyze stockout patterns and their impact on sales",
            inputSchema={
                "type": "object",
                "properties": {
                    "days_back": {
                        "type": "integer",
                        "description": "Number of days to analyze (default: 30)",
                        "default": STOCKOUT_ANALYSIS_DAYS
                    }
                }
            }
        ),
        mcp_types.Tool(
            name="calculate_inventory_turnover",
            description="Calculate inventory turnover metrics for products",
            inputSchema={
                "type": "object",
                "properties": {
                    "product_id": {
                        "type": "string",
                        "description": "Optional specific product ID to analyze"
                    }
                }
            }
        ),
        mcp_types.Tool(
            name="assess_supply_chain_health",
            description="Assess overall supply chain health and identify constraints",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        mcp_types.Tool(
            name="detect_stockout_events",
            description="Detect current and potential stockout events with impact analysis",
            inputSchema={
                "type": "object",
                "properties": {
                    "product_id": {
                        "type": "string",
                        "description": "Optional specific product ID to analyze"
                    },
                    "store_id": {
                        "type": "string",
                        "description": "Optional specific store ID to analyze"
                    }
                }
            }
        ),
        mcp_types.Tool(
            name="monitor_supply_chain_constraints",
            description="Monitor supply chain health and identify operational constraints",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        mcp_types.Tool(
            name="generate_inventory_optimization_recommendations",
            description="Generate inventory optimization recommendations based on turnover and stockout analysis",
            inputSchema={
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "Optional category filter for focused recommendations"
                    }
                }
            }
        ),
        mcp_types.Tool(
            name="generate_forecasting_context",
            description="Generate comprehensive inventory context for demand forecasting models",
            inputSchema={
                "type": "object",
                "properties": {
                    "product_id": {
                        "type": "string",
                        "description": "Optional specific product ID to analyze"
                    },
                    "store_id": {
                        "type": "string",
                        "description": "Optional specific store ID to analyze"
                    },
                    "days_back": {
                        "type": "integer",
                        "description": "Number of days to analyze (default: 90)",
                        "default": FORECASTING_CONTEXT_DAYS
                    }
                }
            }
        ),
        mcp_types.Tool(
            name="integrate_sales_inventory_data",
            description="Create integrated analysis combining sales and inventory data for comprehensive insights",
            inputSchema={
                "type": "object",
                "properties": {
                    "days_back": {
                        "type": "integer",
                        "description": "Number of days to analyze (default: 30)",
                        "default": 30
                    }
                }
            }
        ),
        mcp_types.Tool(
            name="analyze_lead_times_and_optimize_reorder_points",
            description="Analyze supplier lead times and optimize reorder points for improved inventory management",
            inputSchema={
                "type": "object",
                "properties": {
                    "days_back": {
                        "type": "integer",
                        "description": "Number of days to analyze (default: 60)",
                        "default": LEAD_TIME_ANALYSIS_DAYS
                    }
                }
            }
        )
    ]


@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> List[mcp_types.TextContent]:
    """Handle tool calls for inventory management operations."""
    try:
        if name == "get_current_inventory":
            location = arguments.get("location")
            result = get_current_inventory(location)
            
        elif name == "analyze_stockout_patterns":
            days_back = arguments.get("days_back", STOCKOUT_ANALYSIS_DAYS)
            result = analyze_stockout_patterns(days_back)
            
        elif name == "calculate_inventory_turnover":
            product_id = arguments.get("product_id")
            result = calculate_inventory_turnover(product_id)
            
        elif name == "assess_supply_chain_health":
            result = assess_supply_chain_health()
            
        elif name == "detect_stockout_events":
            product_id = arguments.get("product_id")
            store_id = arguments.get("store_id")
            result = detect_stockout_events(product_id, store_id)
            
        elif name == "monitor_supply_chain_constraints":
            result = monitor_supply_chain_constraints()
            
        elif name == "generate_inventory_optimization_recommendations":
            category = arguments.get("category")
            result = generate_inventory_optimization_recommendations(category)
            
        elif name == "generate_forecasting_context":
            product_id = arguments.get("product_id")
            store_id = arguments.get("store_id")
            days_back = arguments.get("days_back", FORECASTING_CONTEXT_DAYS)
            result = generate_forecasting_context(product_id, store_id, days_back)
            
        elif name == "integrate_sales_inventory_data":
            days_back = arguments.get("days_back", 30)
            result = integrate_sales_inventory_data(days_back)
            
        elif name == "analyze_lead_times_and_optimize_reorder_points":
            days_back = arguments.get("days_back", LEAD_TIME_ANALYSIS_DAYS)
            result = analyze_lead_times_and_optimize_reorder_points(days_back)
            
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
        logging.error(f"Error in handle_call_tool for {name}: {e}")
        return [mcp_types.TextContent(
            type="text",
            text=json.dumps({
                "success": False,
                "message": f"Error executing {name}: {str(e)}"
            }, indent=2)
        )]


async def main():
    """Main function to run the Inventory MCP Server."""
    # Run the server using stdin/stdout streams
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="inventory-mcp-server",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


if __name__ == "__main__":
    asyncio.run(main())

@server.list_tools()
async def handle_list_tools() -> List[mcp_types.Tool]:
    """List available inventory management tools."""
    return [
        mcp_types.Tool(
            name="get_current_inventory",
            description="Get current inventory snapshot across all locations or specific location",
            inputSchema={
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "Optional location filter (store location or region)"
                    }
                }
            }
        ),
        mcp_types.Tool(
            name="analyze_stockout_patterns",
            description="Analyze stockout patterns and their impact on sales",
            inputSchema={
                "type": "object",
                "properties": {
                    "days_back": {
                        "type": "integer",
                        "description": "Number of days to analyze (default: 30)",
                        "default": STOCKOUT_ANALYSIS_DAYS
                    }
                }
            }
        ),
        mcp_types.Tool(
            name="calculate_inventory_turnover",
            description="Calculate inventory turnover metrics for products",
            inputSchema={
                "type": "object",
                "properties": {
                    "product_id": {
                        "type": "string",
                        "description": "Optional specific product ID to analyze"
                    }
                }
            }
        ),
        mcp_types.Tool(
            name="assess_supply_chain_health",
            description="Assess overall supply chain health and identify constraints",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        )
    ]


@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> List[mcp_types.TextContent]:
    """Handle tool calls for inventory management operations."""
    try:
        if name == "get_current_inventory":
            location = arguments.get("location")
            result = get_current_inventory(location)
            
        elif name == "analyze_stockout_patterns":
            days_back = arguments.get("days_back", STOCKOUT_ANALYSIS_DAYS)
            result = analyze_stockout_patterns(days_back)
            
        elif name == "calculate_inventory_turnover":
            product_id = arguments.get("product_id")
            result = calculate_inventory_turnover(product_id)
            
        elif name == "assess_supply_chain_health":
            result = assess_supply_chain_health()
            
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
                server_name="inventory-mcp-server",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


if __name__ == "__main__":
    asyncio.run(main())