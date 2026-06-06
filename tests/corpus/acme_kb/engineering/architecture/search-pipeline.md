# Search Pipeline

The search pipeline parses documents into chunks, embeds each chunk, and ranks
results by relevance. A reranking stage reorders the top candidates before they
are returned. The pipeline writes vectors into the vector-store described nearby.
