"""Carnot MCP server package -- exposes EBM verification tools via MCP protocol.

**Researcher summary:**
    This package provides a hardened Model Context Protocol (MCP) server that
    exposes Carnot's verification, constraint extraction, and verify-repair
    pipeline as tools callable by Claude Code or any MCP-compatible client.

**Detailed explanation for engineers:**
    The MCP server is the bridge between LLM agents (like Claude Code) and
    Carnot's verification engine. It wraps the underlying Python APIs with
    production safeguards: execution timeouts, input size limits, and
    structured error responses.

    Entry point: ``python -m carnot.mcp`` runs the server over stdio.

Spec: REQ-CODE-001, REQ-CODE-006, REQ-VERIFY-001
"""

from __future__ import annotations

__all__ = ["create_server"]


def create_server() -> "FastMCP":
    """Create and return the configured MCP server instance.

    **Detailed explanation for engineers:**
        Factory function that imports and returns the FastMCP server from
        the server module. This allows callers to get the server without
        triggering all imports at module level, which is useful for testing
        and for cases where the caller wants to customize before running.

    Returns:
        Configured FastMCP server instance with all tools registered.
    """
    from carnot.mcp.server import mcp_server

    return mcp_server
