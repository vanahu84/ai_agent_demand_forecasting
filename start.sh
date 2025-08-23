#!/bin/bash
set -e  # exit on any error

echo "🔧 Starting MCP servers..."
# Launch drift detection MCP server in the background
python -m autonomous_demand_forecasting.drift_detection_mcp_server &
MCP_PID=$!

# You can add more MCP servers if needed, each with &
# python -m autonomous_demand_forecasting.sales_data_mcp_server &

# Give MCP servers a brief moment to start
sleep 2

echo "🌐 Starting ADK web service..."
exec adk web --host 0.0.0.0 --port "${PORT:-10000}"
