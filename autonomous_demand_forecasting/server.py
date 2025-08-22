import asyncio
import json
import logging  # Added logging
import os
import sqlite3  # For database operations

import mcp.server.stdio  # For running as a stdio server
from dotenv import load_dotenv

# MCP Server Imports
from mcp import types as mcp_types  # Use alias to avoid conflict
from mcp.server.lowlevel import NotificationOptions, Server
from mcp.server.models import InitializationOptions

load_dotenv()

# --- Logging Setup ---
# LOG_FILE_PATH = os.getenv("MCP_LOG_PATH", "/app/telco_network_optimizer_agent/mcp_server_activity.log")
# logging.basicConfig(
#     level=logging.DEBUG,
#     format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
#     handlers=[
#         logging.FileHandler(LOG_FILE_PATH, mode="w"),
#         logging.StreamHandler(),  # Add this line to enable console output
#     ],
# )
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Console output only
    ]
)
# --- End Logging Setup ---

DATABASE_PATH = os.path.join(os.path.dirname(__file__), "autonomous_demand_forecasting.db")


# --- Database Utility Functions ---
def get_db_connection():
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row  # To access columns by name
    return conn


def list_db_tables(dummy_param: str) -> dict:
    """Lists all tables in the SQLite database.

    Args:
        dummy_param (str): This parameter is not used by the function
                           but helps ensure schema generation. A non-empty string is expected.
    Returns:
        dict: A dictionary with keys 'success' (bool), 'message' (str),
              and 'tables' (list[str]) containing the table names if successful.
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        conn.close()
        return {
            "success": True,
            "message": "Tables listed successfully.",
            "tables": tables,
        }
    except sqlite3.Error as e:
        return {"success": False, "message": f"Error listing tables: {e}", "tables": []}
    except Exception as e:  # Catch any other unexpected errors
        return {
            "success": False,
            "message": f"An unexpected error occurred while listing tables: {e}",
            "tables": [],
        }


def get_table_schema(table_name: str) -> dict:
    """Gets the schema (column names and types) of a specific table."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info('{table_name}');")  # Use PRAGMA for schema
    schema_info = cursor.fetchall()
    conn.close()
    if not schema_info:
        raise ValueError(f"Table '{table_name}' not found or no schema information.")

    columns = [{"name": row["name"], "type": row["type"]} for row in schema_info]
    return {"table_name": table_name, "columns": columns}


def query_db_table(table_name: str, columns: str, condition: str) -> list[dict]:
    """Queries a table with an optional condition.

    Args:
        table_name: The name of the table to query.
        columns: Comma-separated list of columns to retrieve (e.g., "id, name"). Defaults to "*".
        condition: Optional SQL WHERE clause condition (e.g., "id = 1" or "completed = 0").
    Returns:
        A list of dictionaries, where each dictionary represents a row.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    query = f"SELECT {columns} FROM {table_name}"
    if condition:
        query += f" WHERE {condition}"
    query += ";"

    try:
        cursor.execute(query)
        results = [dict(row) for row in cursor.fetchall()]
    except sqlite3.Error as e:
        conn.close()
        raise ValueError(f"Error querying table '{table_name}': {e}")
    conn.close()
    return results


def insert_data(table_name: str, data: dict) -> dict:
    """Inserts a new row of data into the specified table.

    Args:
        table_name (str): The name of the table to insert data into.
        data (dict): A dictionary where keys are column names and values are the
                     corresponding values for the new row.

    Returns:
        dict: A dictionary with keys 'success' (bool) and 'message' (str).
              If successful, 'message' includes the ID of the newly inserted row.
    """
    if not data:
        return {"success": False, "message": "No data provided for insertion."}

    conn = get_db_connection()
    cursor = conn.cursor()

    columns = ", ".join(data.keys())
    placeholders = ", ".join(["?" for _ in data])
    values = tuple(data.values())

    query = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"

    try:
        cursor.execute(query, values)
        conn.commit()
        last_row_id = cursor.lastrowid
        return {
            "success": True,
            "message": f"Data inserted successfully. Row ID: {last_row_id}",
            "row_id": last_row_id,
        }
    except sqlite3.Error as e:
        conn.rollback()  # Roll back changes on error
        return {
            "success": False,
            "message": f"Error inserting data into table '{table_name}': {e}",
        }
    finally:
        conn.close()


def delete_data(table_name: str, condition: str) -> dict:
    """Deletes rows from a table based on a given SQL WHERE clause condition.

    Args:
        table_name (str): The name of the table to delete data from.
        condition (str): The SQL WHERE clause condition to specify which rows to delete.
                         This condition MUST NOT be empty to prevent accidental mass deletion.

    Returns:
        dict: A dictionary with keys 'success' (bool) and 'message' (str).
              If successful, 'message' includes the count of deleted rows.
    """
    if not condition or not condition.strip():
        return {
            "success": False,
            "message": "Deletion condition cannot be empty. This is a safety measure to prevent accidental deletion of all rows.",
        }

    conn = get_db_connection()
    cursor = conn.cursor()

    query = f"DELETE FROM {table_name} WHERE {condition}"

    try:
        cursor.execute(query)
        rows_deleted = cursor.rowcount
        conn.commit()
        return {
            "success": True,
            "message": f"{rows_deleted} row(s) deleted successfully from table '{table_name}'.",
            "rows_deleted": rows_deleted,
        }
    except sqlite3.Error as e:
        conn.rollback()
        return {
            "success": False,
            "message": f"Error deleting data from table '{table_name}': {e}",
        }
    finally:
        conn.close()


# --- MCP Server Setup ---
logging.info("Creating SQLite DB MCP Server instance...")
app = Server("sqlite-db-mcp-server")

@app.list_tools()
async def list_mcp_tools() -> list[mcp_types.Tool]:
    """MCP handler to list tools this server exposes."""
    logging.info("SQLite DB MCP Server: Received list_tools request.")
    
    tools = [
        mcp_types.Tool(
            name="list_db_tables",
            description="Lists all tables in the SQLite database",
            inputSchema={
                "type": "object",
                "properties": {
                    "dummy_param": {
                        "type": "string",
                        "description": "This parameter is not used by the function but helps ensure schema generation. A non-empty string is expected."
                    }
                },
                "required": ["dummy_param"]
            }
        ),
        mcp_types.Tool(
            name="get_table_schema",
            description="Gets the schema (column names and types) of a specific table",
            inputSchema={
                "type": "object",
                "properties": {
                    "table_name": {
                        "type": "string",
                        "description": "The name of the table to get schema for"
                    }
                },
                "required": ["table_name"]
            }
        ),
        mcp_types.Tool(
            name="query_db_table",
            description="Queries a table with an optional condition",
            inputSchema={
                "type": "object",
                "properties": {
                    "table_name": {
                        "type": "string",
                        "description": "The name of the table to query"
                    },
                    "columns": {
                        "type": "string",
                        "description": "Comma-separated list of columns to retrieve (e.g., 'id, name'). Defaults to '*'",
                        "default": "*"
                    },
                    "condition": {
                        "type": "string",
                        "description": "Optional SQL WHERE clause condition (e.g., 'id = 1' or 'completed = 0')",
                        "default": ""
                    }
                },
                "required": ["table_name", "columns"]
            }
        ),
        mcp_types.Tool(
            name="insert_data",
            description="Inserts a new row of data into the specified table",
            inputSchema={
                "type": "object",
                "properties": {
                    "table_name": {
                        "type": "string",
                        "description": "The name of the table to insert data into"
                    },
                    "data": {
                        "type": "object",
                        "description": "A dictionary where keys are column names and values are the corresponding values for the new row"
                    }
                },
                "required": ["table_name", "data"]
            }
        ),
        mcp_types.Tool(
            name="delete_data",
            description="Deletes rows from a table based on a given SQL WHERE clause condition",
            inputSchema={
                "type": "object",
                "properties": {
                    "table_name": {
                        "type": "string",
                        "description": "The name of the table to delete data from"
                    },
                    "condition": {
                        "type": "string",
                        "description": "The SQL WHERE clause condition to specify which rows to delete. This condition MUST NOT be empty to prevent accidental mass deletion"
                    }
                },
                "required": ["table_name", "condition"]
            }
        )
    ]
    
    return tools

@app.call_tool()
async def call_mcp_tool(name: str, arguments: dict) -> list[mcp_types.TextContent]:
    """MCP handler to execute a tool call requested by an MCP client."""
    logging.info(f"SQLite DB MCP Server: Received call_tool request for '{name}' with args: {arguments}")

    try:
        if name == "list_db_tables":
            result = list_db_tables(arguments.get("dummy_param", ""))
        elif name == "get_table_schema":
            result = get_table_schema(arguments.get("table_name"))
        elif name == "query_db_table":
            result = query_db_table(
                arguments.get("table_name"),
                arguments.get("columns", "*"),
                arguments.get("condition", "")
            )
        elif name == "insert_data":
            result = insert_data(
                arguments.get("table_name"),
                arguments.get("data", {})
            )
        elif name == "delete_data":
            result = delete_data(
                arguments.get("table_name"),
                arguments.get("condition", "")
            )
        else:
            result = {
                "success": False,
                "message": f"Tool '{name}' not implemented by this server.",
                "available_tools": ["list_db_tables", "get_table_schema", "query_db_table", "insert_data", "delete_data"]
            }
        
        logging.info(f"SQLite DB MCP Server: Tool '{name}' executed successfully")
        response_text = json.dumps(result, indent=2)
        return [mcp_types.TextContent(type="text", text=response_text)]

    except Exception as e:
        logging.error(f"SQLite DB MCP Server: Error executing tool '{name}': {e}", exc_info=True)
        error_payload = {
            "success": False,
            "message": f"Failed to execute tool '{name}': {str(e)}",
            "tool_name": name
        }
        error_text = json.dumps(error_payload)
        return [mcp_types.TextContent(type="text", text=error_text)]


# --- MCP Server Runner ---
async def run_mcp_stdio_server():
    """Runs the MCP server, listening for connections over standard input/output."""
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        logging.info("SQLite DB MCP Stdio Server: Starting handshake with client...")
        await app.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name=app.name,
                server_version="0.1.0",
                capabilities=app.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )
        logging.info("SQLite DB MCP Stdio Server: Run loop finished or client disconnected.")


if __name__ == "__main__":
    logging.info("Launching SQLite DB MCP Server via stdio...")
    try:
        asyncio.run(run_mcp_stdio_server())
    except KeyboardInterrupt:
        logging.info("\nSQLite DB MCP Server (stdio) stopped by user.")
    except Exception as e:
        logging.critical(f"SQLite DB MCP Server (stdio) encountered an unhandled error: {e}", exc_info=True)
    finally:
        logging.info("SQLite DB MCP Server (stdio) process exiting.")
