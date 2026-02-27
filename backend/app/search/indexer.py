"""Phase 2: In-memory Tantivy BM25 index builder.

Builds a Rust-backed inverted index purely in RAM — no disk I/O.
"""

import json
import logging

import tantivy

logger = logging.getLogger("lucio.indexer")


def build_index(chunks: list[dict]) -> tantivy.Index:
    """Build a Tantivy in-memory BM25 index from extracted chunks.

    Schema:
        - chunk_id: stored text
        - text: stored + positional indexed text (BM25 searchable)
        - filename: stored text
        - page_nums: stored JSON field

    Args:
        chunks: List of chunk dicts from Phase 1 extraction.

    Returns:
        A Tantivy Index (already reloaded, ready for searching).
    """
    # Define schema
    builder = tantivy.SchemaBuilder()
    builder.add_text_field("chunk_id", stored=True)
    builder.add_text_field("text", stored=True, index_option="position")
    builder.add_text_field("content", stored=True)  # raw text for embedding
    builder.add_text_field("filename", stored=True)
    builder.add_text_field("page_nums", stored=True)
    schema = builder.build()

    # Create RAM-only index
    index = tantivy.Index(schema)
    writer = index.writer()

    # Populate
    for chunk in chunks:
        writer.add_document(
            tantivy.Document(
                chunk_id=[chunk["chunk_id"]],
                text=[chunk["text"]],
                content=[chunk["content"]],
                filename=[chunk["filename"]],
                page_nums=[json.dumps(chunk["page_nums"])],
            )
        )

    writer.commit()
    index.reload()  # MUST reload before searching

    logger.info(f"Tantivy index built: {len(chunks)} chunks indexed")
    return index
