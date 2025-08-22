import asyncio
import sys
import json
from pathlib import Path

from google.adk.agents import LlmAgent
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StdioServerParameters
try:
    from autonomous_demand_forecasting.ds_prompt import DATA_SCIENTIST_AGENT_PROMPT
except ImportError:
    from ds_prompt import DATA_SCIENTIST_AGENT_PROMPT

# Fix for Windows subprocess support
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# Load JSON config
CONFIG_PATH = Path(__file__).resolve().parent / "mcp_config.json"
with open(CONFIG_PATH, "r") as f:
    mcp_config = json.load(f)["mcpServers"]

# Create a list of MCPToolset instances for each server
toolsets = []
for name, config in mcp_config.items():
    # Skip disabled servers
    if config.get("disabled", False):
        print(f"Skipping disabled MCP Server: {name}")
        continue
        
    print(f"Adding MCP Server: {name}")
    try:
        toolset = MCPToolset(
            connection_params=StdioServerParameters(
                command=config["command"],
                args=config["args"]
            )
        )
        print(f"→ Toolset created for {name}: {toolset}")
        toolsets.append(toolset)
    except Exception as e:
        print(f"✗ Failed to create toolset for {name}: {e}")
        continue

# If no toolsets were successfully created, use empty list
if not toolsets:
    print("⚠ No MCP servers available, creating agent without tools")

# Build the root agent with all MCP servers
root_agent = LlmAgent(
    model="gemini-2.0-flash",
    name="data_scientist_agent",
    instruction=DATA_SCIENTIST_AGENT_PROMPT,
    tools=toolsets  # Multiple MCP servers
)