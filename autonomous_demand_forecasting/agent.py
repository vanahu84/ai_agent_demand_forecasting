import asyncio
import sys
import sys
import json
import time
import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any

from google.adk.agents import LlmAgent
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StdioServerParameters

try:
    from autonomous_demand_forecasting.ds_prompt import DATA_SCIENTIST_AGENT_PROMPT
except ImportError:
    from ds_prompt import DATA_SCIENTIST_AGENT_PROMPT

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Fix for Windows subprocess support
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

class OptimizedMCPManager:
    """Manages MCP server connections with connection pooling and sequential initialization."""
    
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.toolsets: List[MCPToolset] = []
        self.failed_servers: List[str] = []
        
    def load_config(self) -> Dict[str, Any]:
        """Load MCP server configuration."""
        with open(self.config_path, "r") as f:
            return json.load(f)["mcpServers"]
    
    async def initialize_server_sequential(self, name: str, config: Dict[str, Any]) -> MCPToolset:
        """Initialize a single MCP server with timeout handling."""
        if config.get("disabled", False):
            logger.info(f"Skipping disabled MCP Server: {name}")
            return None
        
        logger.info(f"Initializing MCP Server: {name}")
        start_time = time.time()
        
        try:
            # Create toolset with timeout
            toolset = MCPToolset(
                connection_params=StdioServerParameters(
                    command=config["command"],
                    args=config["args"]
                )
            )
            
            # Test the connection by trying to get tools (with timeout)
            try:
                # Give each server more time to initialize
                await asyncio.wait_for(
                    self._test_toolset_connection(toolset), 
                    timeout=15.0  # 15 second timeout per server
                )
                
                elapsed = time.time() - start_time
                logger.info(f"✅ {name} initialized successfully in {elapsed:.2f}s")
                return toolset
                
            except asyncio.TimeoutError:
                logger.warning(f"⏰ {name} initialization timed out after 15s")
                self.failed_servers.append(f"{name} (timeout)")
                return None
                
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"❌ {name} failed to initialize after {elapsed:.2f}s: {e}")
            self.failed_servers.append(f"{name} (error: {str(e)[:50]})")
            return None
    
    async def _test_toolset_connection(self, toolset: MCPToolset):
        """Test if a toolset can connect and get tools."""
        try:
            # Try to get tools to verify the connection works
            tools = await toolset.get_tools()
            logger.debug(f"Toolset connection test passed, found {len(tools) if tools else 0} tools")
        except Exception as e:
            logger.debug(f"Toolset connection test failed: {e}")
            raise
    
    async def initialize_all_servers(self) -> List[MCPToolset]:
        """Initialize all enabled MCP servers sequentially."""
        logger.info("🚀 Starting optimized MCP server initialization")
        
        mcp_config = self.load_config()
        successful_toolsets = []
        
        # Initialize servers one by one to avoid resource contention
        for name, config in mcp_config.items():
            if config.get("disabled", False):
                continue
                
            # Add a small delay between server initializations
            if successful_toolsets:
                await asyncio.sleep(1.0)  # 1 second delay between servers
            
            toolset = await self.initialize_server_sequential(name, config)
            if toolset:
                successful_toolsets.append(toolset)
            
            # Log progress
            logger.info(f"Progress: {len(successful_toolsets)} successful, {len(self.failed_servers)} failed")
        
        # Summary
        logger.info(f"🎯 MCP Server Initialization Complete:")
        logger.info(f"   ✅ Successful: {len(successful_toolsets)}")
        logger.info(f"   ❌ Failed: {len(self.failed_servers)}")
        
        if self.failed_servers:
            logger.warning(f"   Failed servers: {', '.join(self.failed_servers)}")
        
        self.toolsets = successful_toolsets
        return successful_toolsets

async def create_optimized_agent() -> LlmAgent:
    """Create an optimized agent with connection pooling and sequential MCP initialization."""
    
    # Initialize connection pool first
    logger.info("🔧 Initializing database connection pool...")
    try:
        from autonomous_demand_forecasting.database.connection_pool import get_connection_pool
        pool = get_connection_pool()
        stats = pool.get_stats()
        logger.info(f"✅ Connection pool ready: {stats}")
    except Exception as e:
        logger.warning(f"⚠️ Connection pool initialization failed: {e}")
    
    # Initialize MCP servers
    config_path = Path(__file__).resolve().parent / "mcp_config.json"
    mcp_manager = OptimizedMCPManager(config_path)
    
    toolsets = await mcp_manager.initialize_all_servers()
    
    # Create agent with available toolsets
    logger.info("🤖 Creating Data Scientist Agent...")
    
    agent = LlmAgent(
        model="gemini-2.0-flash",
        name="optimized_data_scientist_agent",
        instruction=DATA_SCIENTIST_AGENT_PROMPT,
        tools=toolsets if toolsets else []  # Use empty list if no tools available
    )
    
    logger.info(f"✅ Agent created with {len(toolsets)} MCP toolsets")
    
    # Log available capabilities
    if toolsets:
        logger.info("🔧 Available MCP capabilities:")
        for i, toolset in enumerate(toolsets, 1):
            logger.info(f"   {i}. {type(toolset).__name__}")
    else:
        logger.warning("⚠️ Agent created without MCP tools - will have limited functionality")
    
    return agent

# Create the optimized agent
async def main():
    """Main function to create and test the optimized agent."""
    try:
        agent = await create_optimized_agent()
        logger.info("🎉 Optimized agent ready for use!")
        return agent
    except Exception as e:
        logger.error(f"❌ Failed to create optimized agent: {e}")
        raise

# For compatibility with the existing agent.py interface
if __name__ == "__main__":
    # If run directly, create and test the agent
    asyncio.run(main())
else:
    # If imported, create the agent synchronously for ADK using simple approach like telco agent
    logger.info("🚀 Initializing Demand Forecasting Agent with MCP servers")
    
    try:
        # Load MCP configuration
        config_path = Path(__file__).resolve().parent / "mcp_config.json"
        with open(config_path, "r") as f:
            mcp_config = json.load(f)["mcpServers"]
        
        # Create toolsets for enabled servers only (simple approach)
        toolsets = []
        failed_servers = []
        
        for name, config in mcp_config.items():
            if config.get("disabled", False):
                logger.info(f"Skipping disabled MCP Server: {name}")
                continue
                
            logger.info(f"🔧 Adding MCP Server: {name}")
            try:
                toolset = MCPToolset(
                    connection_params=StdioServerParameters(
                        command=config["command"],
                        args=config["args"]
                    )
                )
                toolsets.append(toolset)
                logger.info(f"✅ Toolset created for {name}: {type(toolset).__name__}")
            except Exception as e:
                logger.error(f"❌ Failed to create toolset for {name}: {e}")
                failed_servers.append(name)
        
        # Create the agent with available toolsets
        root_agent = LlmAgent(
            model="gemini-2.0-flash",
            name="optimized_data_scientist_agent",
            instruction=DATA_SCIENTIST_AGENT_PROMPT,
            tools=toolsets
        )
        
        logger.info(f"✅ Demand Forecasting Agent initialized with {len(toolsets)} MCP toolsets")
        
        if failed_servers:
            logger.warning(f"⚠️ Some MCP servers failed to initialize: {failed_servers}")
        
    except Exception as e:
        logger.error(f"❌ Failed to create optimized agent during import: {e}")
        # Fallback to basic agent without MCP tools
        root_agent = LlmAgent(
            model="gemini-2.0-flash",
            name="fallback_data_scientist_agent",
            instruction=DATA_SCIENTIST_AGENT_PROMPT,
            tools=[]
        )
        logger.warning("⚠️ Created fallback agent without MCP tools")