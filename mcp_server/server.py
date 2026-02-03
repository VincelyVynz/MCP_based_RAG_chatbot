from fastmcp import server

import mcp_server.tools.retrieve
import mcp_server.tools.update

app = server.FastMCP(name = "RAG chatbot with MCP")

if __name__ == "__main__":
    app.run()