#!/bin/bash
set -e

echo "🔧 Starting MCP servers..."
# Start Drift Detection MCP server in background
python -m autonomous_demand_forecasting.drift_detection_mcp_server &
MCP_PID=$!

# Trap to clean up MCP process when container exits
trap "kill $MCP_PID" EXIT

# Optional: If you have more MCP servers, start them similarly:
# python -m autonomous_demand_forecasting.sales_data_mcp_server & 
# OTHER_PID=$!
# trap "kill $MCP_PID $OTHER_PID" EXIT

# Give servers a moment to boot
sleep 2

echo "🌐 Starting ADK web service..."
exec adk web --host 0.0.0.0 --port "${PORT:-10000}"
