from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
RELEASE_DIR = BASE_DIR / "data" / "releases" / "real_dft_campaign_v1"


class ReleaseManifestTests(unittest.TestCase):
    def test_manifest_files_match_checksums(self) -> None:
        manifest_path = RELEASE_DIR / "release_manifest.json"
        self.assertTrue(manifest_path.exists(), "release manifest missing")
        manifest = json.loads(manifest_path.read_text())
        self.assertGreater(len(manifest), 0)

        for entry in manifest:
            path = RELEASE_DIR / entry["path"]
            self.assertTrue(path.exists(), f"missing {entry['path']}")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(digest, entry["sha256"], f"checksum mismatch {entry['path']}")


if __name__ == "__main__":
    unittest.main()
