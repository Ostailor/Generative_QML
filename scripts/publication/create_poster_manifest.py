#!/usr/bin/env python3
"""Create poster package manifest with checksums."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
POSTER_DIR = BASE_DIR / "docs" / "poster"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    entries = []
    for path in sorted(p for p in POSTER_DIR.rglob("*") if p.is_file() and p.name != "poster_manifest.json"):
        rel = path.relative_to(POSTER_DIR)
        entries.append({"path": str(rel), "sha256": sha256(path), "size_bytes": path.stat().st_size})
    out = {"version": "1.0.0", "entries": entries}
    (POSTER_DIR / "poster_manifest.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({"entries": len(entries), "manifest": str(POSTER_DIR / 'poster_manifest.json')}, indent=2))


if __name__ == "__main__":
    main()
