#!/usr/bin/env python3
"""Build a ~1GB stress corpus by duplicating existing documents.

Creates unique filenames per copy so each doc produces distinct chunk_ids
in the pipeline (fetcher.py uses os.path.basename → must be flat, no subdirs).

Usage:
    python eval/build_stress_corpus.py                      # default: ~1GB
    python eval/build_stress_corpus.py --target-mb 500      # custom size
    python eval/build_stress_corpus.py --source /path/to.zip --output /tmp/stress.zip
"""

import argparse
import io
import json
import os
import sys
import zipfile
from pathlib import Path

EVAL_DIR = Path(__file__).parent
GROUND_TRUTH = EVAL_DIR / "ground_truth.json"
DEFAULT_OUTPUT = EVAL_DIR / "stress_corpus.zip"

# Mirror fetcher.py filtering logic
ALLOWED_EXTENSIONS = {".pdf", ".docx"}
SKIP_PREFIXES = ("__MACOSX/", "._")


def get_valid_entries(zf: zipfile.ZipFile) -> list[tuple[str, bytes]]:
    """Extract valid (basename, bytes) entries matching fetcher.py logic."""
    entries = []
    for name in zf.namelist():
        if name.endswith("/"):
            continue
        basename = os.path.basename(name)
        if basename.startswith(".") or any(
            name.startswith(p) for p in SKIP_PREFIXES
        ):
            continue
        _, ext = os.path.splitext(name.lower())
        if ext not in ALLOWED_EXTENSIONS:
            continue
        entries.append((basename, zf.read(name)))
    return entries


def build_stress_corpus(source_path: str, output_path: str, target_mb: int):
    """Build stress corpus zip by duplicating source docs."""
    target_bytes = target_mb * 1024 * 1024

    # Read source zip
    with zipfile.ZipFile(source_path, "r") as zf:
        entries = get_valid_entries(zf)

    if not entries:
        print(f"No valid documents found in {source_path}")
        sys.exit(1)

    source_size = sum(len(data) for _, data in entries)
    print(f"Source: {len(entries)} docs, {source_size / 1024 / 1024:.1f}MB uncompressed")
    print(f"Target: ~{target_mb}MB")

    # Calculate passes needed
    passes_needed = max(1, (target_bytes // source_size) + 1)
    print(f"Will create {passes_needed} copies ({passes_needed} x {len(entries)} = {passes_needed * len(entries)} files)")

    # Build output zip
    buf = io.BytesIO()
    total_written = 0
    total_files = 0

    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as out_zf:
        for copy_idx in range(passes_needed):
            for basename, data in entries:
                if copy_idx == 0:
                    # Copy 0: original filenames (existing eval questions still work)
                    out_name = basename
                else:
                    # Copy N: c{N}_{basename} → unique chunk_ids
                    out_name = f"c{copy_idx}_{basename}"

                out_zf.writestr(out_name, data)
                total_written += len(data)
                total_files += 1

            pct = (total_written / target_bytes) * 100
            print(
                f"  Pass {copy_idx + 1}/{passes_needed}: "
                f"{total_files} files, "
                f"{total_written / 1024 / 1024:.0f}MB ({pct:.0f}%)"
            )

            if total_written >= target_bytes:
                break

    # Write output
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(buf.getvalue())

    final_size = output.stat().st_size
    print(
        f"\nOutput: {output} "
        f"({final_size / 1024 / 1024:.0f}MB compressed, "
        f"{total_written / 1024 / 1024:.0f}MB uncompressed, "
        f"{total_files} files)"
    )


def main():
    parser = argparse.ArgumentParser(description="Build stress test corpus")
    parser.add_argument(
        "--source",
        default=None,
        help="Source zip path (default: from ground_truth.json)",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help=f"Output zip path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--target-mb",
        type=int,
        default=1024,
        help="Target uncompressed size in MB (default: 1024)",
    )
    args = parser.parse_args()

    # Resolve source
    source = args.source
    if not source:
        with open(GROUND_TRUTH) as f:
            gt = json.load(f)
        source = gt["corpus_url"]
    if not os.path.exists(source):
        print(f"Source zip not found: {source}")
        sys.exit(1)

    build_stress_corpus(source, args.output, args.target_mb)


if __name__ == "__main__":
    main()
