"""Document ingestion: load, chunk, embed, and write into pgvector.

Two decisions worth defending in the dissertation:

1. **Heading-prefixed chunks.** Each chunk is embedded with its document title
   and section heading prepended. A bare chunk such as "5 to 7 business days"
   is nearly unretrievable on its own; "Refund Policy > Refund processing times
   > 5 to 7 business days" is not. The prefix is stored in `content` so the
   reranker and the generator see the same context the embedder saw.

2. **Idempotent re-ingestion.** Chunks are keyed on `(source, chunk_index)` and
   upserted, and each source is cleared before reload. Re-running ingestion
   never produces duplicates, which is what makes the evaluation job in CI
   reproducible.

Run:  python -m src.ingestion.ingest --reset
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import get_settings
from src.retrieval.bm25 import refresh_bm25_index
from src.retrieval.embeddings import embed_texts
from src.retrieval.vector_store import (
    clear_source,
    count_chunks,
    init_schema,
    insert_chunks,
)

logger = logging.getLogger(__name__)

_HEADING_PATTERN = re.compile(r"^#{1,6}\s+(.*)$")
_DOC_ID_PATTERN = re.compile(r"^Document ID:\s*(\S+)", re.MULTILINE)


def _build_splitter() -> RecursiveCharacterTextSplitter:
    settings = get_settings()
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        # Prefer splitting on section boundaries, then paragraphs, then lines.
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " "],
        keep_separator=True,
    )


def _extract_headings(text: str) -> tuple[str, str]:
    """Return (document title, first section heading) found in a chunk."""
    title = ""
    section = ""
    for line in text.splitlines():
        match = _HEADING_PATTERN.match(line.strip())
        if not match:
            continue
        heading = match.group(1).strip()
        if line.strip().startswith("# ") and not title:
            title = heading
        elif not section:
            section = heading
    return title, section


def _split_document(path: Path) -> list[dict]:
    raw = path.read_text(encoding="utf-8")
    doc_title, _ = _extract_headings(raw)
    doc_id_match = _DOC_ID_PATTERN.search(raw)
    doc_id = doc_id_match.group(1) if doc_id_match else path.stem

    splitter = _build_splitter()
    pieces = splitter.split_text(raw)

    records: list[dict] = []
    current_section = ""
    for index, piece in enumerate(pieces):
        _, section = _extract_headings(piece)
        if section:
            current_section = section

        breadcrumb = " > ".join(part for part in (doc_title, current_section) if part)
        content = f"[{breadcrumb}]\n{piece.strip()}" if breadcrumb else piece.strip()

        records.append(
            {
                "source": path.name,
                "doc_id": doc_id,
                "chunk_index": index,
                "content": content,
                "metadata": {
                    "doc_title": doc_title,
                    "section": current_section,
                    "char_count": len(piece),
                },
            }
        )
    return records


def ingest_file(path: Path, reset: bool = True) -> int:
    """Ingest a single document. Returns the number of chunks written."""
    records = _split_document(path)
    if not records:
        logger.warning("No chunks produced for %s", path)
        return 0

    embeddings = embed_texts([r["content"] for r in records])
    for record, embedding in zip(records, embeddings, strict=True):
        record["embedding"] = embedding

    if reset:
        removed = clear_source(path.name)
        if removed:
            logger.info("Cleared %d existing chunks for %s", removed, path.name)

    written = insert_chunks(records)
    logger.info("Ingested %s -> %d chunks", path.name, written)
    return written


def ingest_corpus(data_dir: Path | None = None, reset: bool = True) -> dict[str, int]:
    """Ingest every .txt and .md file in the data directory."""
    settings = get_settings()
    directory = data_dir or settings.absolute_data_dir
    if not directory.exists():
        raise FileNotFoundError(f"Data directory not found: {directory}")

    init_schema()

    files = sorted(
        [p for p in directory.iterdir() if p.suffix.lower() in {".txt", ".md"}]
    )
    if not files:
        raise FileNotFoundError(f"No .txt or .md documents in {directory}")

    results = {path.name: ingest_file(path, reset=reset) for path in files}
    refresh_bm25_index()
    logger.info("Corpus ingested: %d chunks total", count_chunks())
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest the RAGuard corpus")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Clear each source before reloading (default behaviour)",
    )
    parser.add_argument(
        "--no-reset", dest="reset", action="store_false", help="Upsert without clearing"
    )
    parser.set_defaults(reset=True)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    results = ingest_corpus(data_dir=args.data_dir, reset=args.reset)
    for source, count in results.items():
        print(f"{source:<32} {count:>4} chunks")
    print(f"{'TOTAL':<32} {sum(results.values()):>4} chunks")


if __name__ == "__main__":
    main()
