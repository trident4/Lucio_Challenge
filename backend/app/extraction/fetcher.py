"""Phase 1a: Smart corpus fetcher — local path or remote URL.

Also handles unzipping into picklable (filename, bytes) tuples
for multiprocessing dispatch.
"""

import io
import logging
import os
import tempfile
import zipfile

import httpx

logger = logging.getLogger("lucio.fetcher")

# Only extract these file types from the zip
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".xlsx"}

# Skip these junk paths
SKIP_PREFIXES = ("__MACOSX/", "._")


async def fetch_corpus(corpus_url: str) -> str:
    """Return a local file path to the corpus zip.

    For local files, returns the path directly (no memory copy).
    For remote URLs, downloads to a temp file on disk.

    Args:
        corpus_url: Local file path or remote HTTP(S) URL.

    Returns:
        Path string usable with zipfile.ZipFile().
    """
    if os.path.exists(corpus_url):
        logger.info(f"Using corpus from local path: {corpus_url}")
        return corpus_url

    logger.info(f"Downloading corpus from: {corpus_url}")
    tmp = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
    try:
        async with httpx.AsyncClient(timeout=300.0, follow_redirects=True) as client:
            async with client.stream("GET", corpus_url) as resp:
                resp.raise_for_status()
                total = int(resp.headers.get("content-length", 0))
                downloaded = 0
                last_log = 0
                async for chunk in resp.aiter_bytes(chunk_size=65536):
                    tmp.write(chunk)
                    downloaded += len(chunk)
                    if total and downloaded - last_log >= 5_000_000:
                        logger.info(
                            f"Download: {downloaded // 1_000_000}/"
                            f"{total // 1_000_000}MB "
                            f"({downloaded * 100 // total}%)"
                        )
                        last_log = downloaded
        tmp.close()
        logger.info(f"Downloaded {downloaded / 1_000_000:.1f}MB to {tmp.name}")
        return tmp.name
    except Exception:
        tmp.close()
        os.unlink(tmp.name)
        raise


def unzip_to_tuples(source: str | io.BytesIO) -> list[tuple[str, bytes]]:
    """Extract zip contents into picklable (filename, raw_bytes) tuples.

    Filters to .pdf/.docx only, skips __MACOSX and dot-files.

    Args:
        source: File path or in-memory BytesIO of the zip.

    Returns:
        List of (filename, file_bytes) tuples ready for multiprocessing.
    """
    tuples = []
    with zipfile.ZipFile(source, "r") as zf:
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
