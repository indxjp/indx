"""Lazy registration table for first-party implementations.

Each entry is an import path resolved only when that slot+name is actually selected, so
importing the registry never drags in a heavy/optional dependency. Built-ins always win
name collisions over third-party plugins (file-architecture §6).
"""

from __future__ import annotations

# slot -> {name -> "module:attr"}
BUILTINS: dict[str, dict[str, str]] = {
    "parser": {
        "plaintext": "indx.parsers.plaintext:PlainTextParser",
        "docling": "indx.parsers.docling:DoclingParser",
        "unstructured": "indx.parsers.unstructured:UnstructuredParser",
        "llamaparse": "indx.parsers.llamaparse:LlamaParseParser",
        "markitdown": "indx.parsers.markitdown:MarkItDownParser",
        "textract": "indx.parsers.textract:TextractParser",
        "docintel": "indx.parsers.docintel:DocumentIntelligenceParser",
        "docai": "indx.parsers.docai:DocumentAIParser",
    },
    "llm": {
        "none": "indx.llm.none:NullLLM",
        "openai": "indx.llm.openai:OpenAILLM",
        "ollama": "indx.llm.ollama:OllamaLLM",
        "anthropic": "indx.llm.anthropic:AnthropicLLM",
        "vllm": "indx.llm.vllm:VLLMClient",
        "azure": "indx.llm.azure:AzureOpenAILLM",
        "bedrock": "indx.llm.bedrock:BedrockLLM",
        "vertex": "indx.llm.vertex:VertexLLM",
        "litellm": "indx.llm.litellm:LiteLLMClient",
    },
    "vlm": {
        "none": "indx.vlm.none:NullVLM",
        "gpt4o": "indx.vlm.gpt4o:GPT4oVLM",
        "qwen-vl": "indx.vlm.qwen_vl:QwenVLClient",
        "local": "indx.vlm.local:LocalVLM",
        "bedrock": "indx.vlm.bedrock:BedrockVLM",
        "azure": "indx.vlm.azure:AzureOpenAIVLM",
        "vertex": "indx.vlm.vertex:VertexVLM",
    },
    "embedder": {
        "hash": "indx.embed.hash_embedder:HashEmbedder",
        "openai": "indx.embed.openai:OpenAIEmbedder",
        "bge-m3": "indx.embed.bge_m3:BGEM3Embedder",
        "e5": "indx.embed.e5:E5Embedder",
        "cohere": "indx.embed.cohere:CohereEmbedder",
        "bedrock": "indx.embed.bedrock:BedrockEmbedder",
        "azure": "indx.embed.azure:AzureOpenAIEmbedder",
        "vertex": "indx.embed.vertex:VertexEmbedder",
        "litellm": "indx.embed.litellm:LiteLLMEmbedder",
    },
    "store": {
        "jsonl": "indx.store.jsonl:JsonlStore",
        "qdrant": "indx.store.qdrant:QdrantStore",
        "pgvector": "indx.store.pgvector:PgVectorStore",
        "chroma": "indx.store.chroma:ChromaStore",
        "lancedb": "indx.store.lancedb:LanceDBStore",
        "s3vectors": "indx.store.s3vectors:S3VectorsStore",
        "opensearch": "indx.store.opensearch:OpenSearchStore",
        "azure-search": "indx.store.azure_search:AzureAISearchStore",
        "bigquery": "indx.store.bigquery:BigQueryStore",
        "vertex-vector": "indx.store.vertex_vector:VertexVectorStore",
    },
    "writer": {
        "indx": "indx.output.indx_writer:IndxWriter",
        "jsonl": "indx.output.jsonl_writer:JsonlWriter",
        "langchain": "indx.output.langchain:LangChainWriter",
        "llamaindex": "indx.output.llamaindex:LlamaIndexWriter",
    },
}

# (slot, name) -> pyproject optional-dependencies extra, for the MissingExtraError fallback
# when a transitive ModuleNotFoundError escapes an adapter that did not gate via require_extra.
# Keyed explicitly because the extra name cannot be derived from the slot name (e.g. the
# ``qwen-vl`` slot maps to the ``qwen-vl`` extra, ``local`` to ``vlm-local``).
EXTRAS: dict[tuple[str, str], str] = {
    ("parser", "docling"): "docling",
    ("parser", "unstructured"): "unstructured",
    ("parser", "llamaparse"): "llamaparse",
    ("parser", "markitdown"): "markitdown",
    ("llm", "openai"): "openai",
    ("llm", "ollama"): "ollama",
    ("llm", "anthropic"): "anthropic",
    ("llm", "vllm"): "vllm",
    ("llm", "azure"): "azure",
    ("llm", "litellm"): "litellm",
    ("embedder", "litellm"): "litellm",
    ("vlm", "gpt4o"): "openai",
    ("vlm", "qwen-vl"): "qwen-vl",
    ("vlm", "local"): "vlm-local",
    ("embedder", "openai"): "openai",
    ("embedder", "bge-m3"): "bge",
    ("embedder", "e5"): "e5",
    ("embedder", "cohere"): "cohere",
    ("store", "qdrant"): "qdrant",
    ("store", "pgvector"): "pgvector",
    ("store", "chroma"): "chroma",
    ("store", "lancedb"): "lancedb",
    ("writer", "langchain"): "langchain",
    ("writer", "llamaindex"): "llamaindex",
    # Cloud profile backends — value is the PROFILE extra (the one-install story), so the
    # MissingExtraError hint points the user at the whole-stack install (file-architecture §6b).
    ("parser", "textract"): "aws",
    ("llm", "bedrock"): "aws",
    ("vlm", "bedrock"): "aws",
    ("embedder", "bedrock"): "aws",
    ("store", "s3vectors"): "aws",
    ("store", "opensearch"): "aws-opensearch",
    ("parser", "docintel"): "azure",
    ("vlm", "azure"): "azure",
    ("embedder", "azure"): "azure",
    ("store", "azure-search"): "azure",
    ("parser", "docai"): "gcp",
    ("llm", "vertex"): "gcp",
    ("vlm", "vertex"): "gcp",
    ("embedder", "vertex"): "gcp",
    ("store", "bigquery"): "gcp",
    ("store", "vertex-vector"): "gcp-vectorsearch",
}

# Entry-point group(s) per slot, for third-party plugin discovery (file-architecture §6).
# A slot may list more than one group: the registry consults each in order so a documented
# group name and a legacy alias both resolve. The writer/output slot accepts the documented
# ``indx.outputs`` group and keeps ``indx.writers`` for back-compat.
ENTRY_POINT_GROUPS: dict[str, tuple[str, ...]] = {
    "parser": ("indx.parsers",),
    "llm": ("indx.llms",),
    "vlm": ("indx.vlms",),
    "embedder": ("indx.embedders",),
    "store": ("indx.stores",),
    "writer": ("indx.outputs", "indx.writers"),
}

# Entry-point group for third-party pipeline stages (registry-and-defaults: the ``stages`` slot).
STAGE_ENTRY_POINT_GROUP = "indx.stages"
