#!/bin/bash
set -e # Exit immediately on error

echo "⚙️ Installing dependencies (if needed)..."
# Optional: only use this line if venv or dependencies are not pre-installed
# pip install -r requirements.txt

echo "🔧 Preparing MCP server environment..."
# Create writable log directory and file if needed
# mkdir -p /tmp/mcp_logs
# touch /tmp/mcp_logs/mcp_server_activity.log
# chmod 755 /tmp/mcp_logs
# chmod 644 /tmp/mcp_logs/mcp_server_activity.log
# chmod 777 /app/telco_network_optimizer_agent/

# Set environment variable for MCP server log path
# export MCP_LOG_PATH="/tmp/mcp_logs/mcp_server_activity.log"

# Test if MCP server can run (don't fail if chmod doesn't work)
echo "🧪 Testing MCP server configuration..."
python /app/telco_network_optimizer_agent/tower_load_mcp_server.py --help > /dev/null 2>&1 || echo "⚠️  MCP server test failed - continuing anyway"

echo "🌐 Starting ADK MCP server..."
adk web --host 0.0.0.0 --port "${PORT:-7860}"
