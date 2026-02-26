"""Phase 1a: Smart corpus fetcher — local path or remote URL.

Also handles unzipping into picklable (filename, bytes) tuples
for multiprocessing dispatch.
"""

import io
import logging
import os
import zipfile

import httpx

logger = logging.getLogger("lucio.fetcher")

# Only extract these file types from the zip
ALLOWED_EXTENSIONS = {".pdf", ".docx"}

# Skip these junk paths
SKIP_PREFIXES = ("__MACOSX/", "._")


async def fetch_corpus(corpus_url: str) -> io.BytesIO:
    """Download or read a corpus zip into an in-memory BytesIO.

    Args:
        corpus_url: Local file path or remote HTTP(S) URL.

    Returns:
        BytesIO containing the raw zip bytes.
    """
    if os.path.exists(corpus_url):
        logger.info(f"Loading corpus from local path: {corpus_url}")
        with open(corpus_url, "rb") as f:
            return io.BytesIO(f.read())

    logger.info(f"Downloading corpus from: {corpus_url}")
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(corpus_url)
        resp.raise_for_status()
        return io.BytesIO(resp.content)


def unzip_to_tuples(zip_bytesio: io.BytesIO) -> list[tuple[str, bytes]]:
    """Extract zip contents into picklable (filename, raw_bytes) tuples.

    Filters to .pdf/.docx only, skips __MACOSX and dot-files.

    Args:
        zip_bytesio: In-memory zip file.

    Returns:
        List of (filename, file_bytes) tuples ready for multiprocessing.
    """
    tuples = []
    with zipfile.ZipFile(zip_bytesio, "r") as zf:
        for name in zf.namelist():
            # Skip directories
            if name.endswith("/"):
                continue
            # Skip junk files
            basename = os.path.basename(name)
            if basename.startswith(".") or any(
                name.startswith(p) for p in SKIP_PREFIXES
            ):
                continue
            # Filter by extension
            _, ext = os.path.splitext(name.lower())
            if ext not in ALLOWED_EXTENSIONS:
                continue

            tuples.append((basename, zf.read(name)))

    logger.info(f"Extracted {len(tuples)} documents from zip")
    return tuples
