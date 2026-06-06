"""indx.agent — plug a knowledge space into any AI agent, USB-drive style.

A ``.indx`` archive is a portable knowledge space. This package is the plug: one call turns
it into tools for whichever agent framework you use.

```python
from indx.agent import connect

kb = connect("ai-ready/handbook.indx")     # load the "USB drive"

kb.langchain()        # LangChain StructuredTools  (+ kb.langchain_retriever())
kb.openai()           # OpenAI Agents SDK function tools
kb.pydantic_ai()      # Pydantic AI tools
kb.claude()           # Claude Agent SDK in-process MCP server
kb.mcp()              # FastMCP server — Mastra & any MCP client
```

Or run it as a standalone MCP server from the shell — ``indx mcp ai-ready/handbook.indx`` —
and connect Claude Desktop, Cursor, or Mastra to it with no Python glue.

Importing this package is safe on a bare ``pip install indx``: the framework adapters are
imported lazily and each gates on its own optional extra (``indx[langchain]``,
``indx[openai-agents]``, ``indx[pydantic-ai]``, ``indx[claude-agent]``, ``indx[mcp]`` — or
``indx[agent]`` for all of them).
"""

from indx.agent.connector import KnowledgeConnector, connect
from indx.agent.schema import (
    GET_DOCUMENT_TOOL,
    OVERVIEW_TOOL,
    SEARCH_TOOL,
    TOOLS,
    DocumentCard,
    DocumentDetail,
    Hit,
    SearchResults,
    SpaceOverview,
    ToolDef,
)

__all__ = [
    "connect",
    "KnowledgeConnector",
    "Hit",
    "SearchResults",
    "DocumentCard",
    "DocumentDetail",
    "SpaceOverview",
    "ToolDef",
    "TOOLS",
    "SEARCH_TOOL",
    "OVERVIEW_TOOL",
    "GET_DOCUMENT_TOOL",
]
