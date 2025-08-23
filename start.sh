#!/bin/bash
set -e  # exit immediately on error

echo "🔧 Starting all MCP servers in background..."

# Start each MCP server as a background process
python -m autonomous_demand_forecasting.drift_detection_mcp_server &
PID_DRIFT=$!

python -m autonomous_demand_forecasting.sales_data_mcp_server &
PID_SALES=$!

python -m autonomous_demand_forecasting.inventory_mcp_server &
PID_INVENTORY=$!

# python -m autonomous_demand_forecasting.forecasting_model_mcp_server &
# PID_FORECAST=$!

# python -m autonomous_demand_forecasting.model_validation_mcp_server &
# PID_VALIDATION=$!

# Trap to clean up all background MCP processes on container exit
trap "kill $PID_DRIFT $PID_SALES $PID_INVENTORY" EXIT


# Optionally give them a moment to start
sleep 2

echo "🌐 Starting ADK web service..."
exec adk web --host 0.0.0.0 --port "${PORT:-10000}"
