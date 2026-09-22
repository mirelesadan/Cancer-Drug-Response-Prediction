"""Retrieve the exact Dataverse files named by PyTDC 1.1.15.

This avoids PyTDC's unrelated import-time dependencies while using the same
file identifiers and URLs as tdc.utils.load.download_wrapper.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import requests


FILES = {
    "gdsc2.pkl": 4165727,
    "gdsc_gene_symbols.tab": 5255026,
}
EXPECTED_MD5 = {
    "gdsc2.pkl": "217ccb2c49dc43485924f8678eaf7e34",
    "gdsc_gene_symbols.tab": "a352ceb1a4586042324c43b357c90cf3",
}
BASE_URL = "https://dataverse.harvard.edu/api/access/datafile/"


def digest_file(path: Path) -> dict[str, object]:
    hashes = {name: hashlib.new(name) for name in ("sha256", "md5")}
    size = 0
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            size += len(block)
            for value in hashes.values():
                value.update(block)
    return {"bytes": size, **{name: value.hexdigest() for name, value in hashes.items()}}


def retrieve(destination: Path, file_id: int) -> dict[str, object]:
    url = f"{BASE_URL}{file_id}"
    metadata_url = f"https://dataverse.harvard.edu/api/files/{file_id}/metadata"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        print(f"Using existing {destination}; verifying checksum", flush=True)
        headers: dict[str, str | None] = {}
    else:
        temporary = destination.with_suffix(destination.suffix + ".part")
        if temporary.exists():
            raise FileExistsError(f"Partial download exists: {temporary}")
        with requests.get(url, stream=True, timeout=(20, 180)) as response:
            response.raise_for_status()
            headers = {
                "content_type": response.headers.get("Content-Type"),
                "content_length": response.headers.get("Content-Length"),
                "last_modified": response.headers.get("Last-Modified"),
                "etag": response.headers.get("ETag"),
            }
            with temporary.open("wb") as output:
                for block in response.iter_content(chunk_size=1024 * 1024):
                    if block:
                        output.write(block)
        temporary.replace(destination)

    metadata: dict[str, object] | None = None
    try:
        response = requests.get(metadata_url, timeout=(20, 30))
        response.raise_for_status()
        metadata = response.json()
    except (requests.RequestException, ValueError) as exc:
        metadata = {"metadata_error": str(exc)}

    checksums = digest_file(destination)
    expected = EXPECTED_MD5[destination.name]
    if checksums["md5"] != expected:
        raise ValueError(f"Checksum mismatch for {destination}: expected MD5 {expected}, got {checksums['md5']}")
    return {
        "source_url": url,
        "metadata_url": metadata_url,
        "downloaded_or_verified_utc": datetime.now(timezone.utc).isoformat(),
        "path": str(destination),
        "checksums": checksums,
        "expected_md5": expected,
        "response_headers": headers,
        "file_metadata": metadata,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    args = parser.parse_args()
    records = {name: retrieve(args.data_dir / name, file_id) for name, file_id in FILES.items()}
    manifest = args.data_dir / "provenance.json"
    manifest.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {manifest}")
    for name, record in records.items():
        print(name, record["checksums"])


if __name__ == "__main__":
    main()
