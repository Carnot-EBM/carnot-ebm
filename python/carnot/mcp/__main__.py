"""Entry point for ``python -m carnot.mcp``.

**Detailed explanation for engineers:**
    Allows running the MCP server as a module: ``python -m carnot.mcp``.
    This is the recommended way to start the server in production, as it
    avoids sys.path manipulation and ensures the carnot package is properly
    installed and importable.

Spec: REQ-CODE-001
"""

from carnot.mcp.server import mcp_server

mcp_server.run(transport="stdio")
