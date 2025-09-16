#!/usr/bin/env python3
"""Dataset ingestion orchestrator for M1 T1.1.

Downloads and validates three primary datasets:
1. Perovskite materials data from the Materials Project REST API.
2. High-entropy alloy property data from Kaggle (NIMS/HEA dataset).
3. Single-atom-alloy (doped nanoparticle) data from Catalysis-Hub.org.

Environment variables (see `.env` template) supply required credentials.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import stat
import sys
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import requests
import yaml

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    load_dotenv = None

try:
    from catalysis_hub import get_reactions
except ImportError:  # pragma: no cover - optional dependency
    get_reactions = None

try:
    from mp_api.client import MPRester
except ImportError:  # pragma: no cover - optional dependency
    MPRester = None

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
METADATA_DIR = BASE_DIR / "data" / "metadata"
MANIFEST_PATH = METADATA_DIR / "provenance_manifest.csv"
SCHEMA_PATH = METADATA_DIR / "schemas.yaml"
DEFAULT_INGESTION_DATE = datetime.utcnow().date().isoformat()
CATALYSIS_HUB_GRAPHQL = "https://api.catalysis-hub.org/graphql"

if load_dotenv is not None:
    env_path = BASE_DIR / ".env"
    if env_path.exists():
        load_dotenv(env_path)
else:
    print("[INFO] python-dotenv not installed; ensure environment variables are exported before running.")

@dataclass
class DatasetConfig:
    domain: str
    version: str
    source: str
    license: str
    target_dir: Path
    expected_columns: List[str]


CONFIGS: Dict[str, DatasetConfig] = {
    "perovskites": DatasetConfig(
        domain="perovskites",
        version="perovskites-v0.1.0",
        source="https://next-gen.materialsproject.org/api/v2/materials",
        license="CC-BY-4.0",
        target_dir=RAW_DIR / "perovskites",
        expected_columns=[
            "material_id",
            "pretty_formula",
            "spacegroup",
            "band_gap",
            "formation_energy_per_atom",
            "e_above_hull",
        ],
    ),
    "high_entropy_alloys": DatasetConfig(
        domain="high_entropy_alloys",
        version="hea-v0.1.0",
        source="https://www.kaggle.com/",  # Detailed slug stored via env
        license="CC-BY-SA-4.0",
        target_dir=RAW_DIR / "high_entropy_alloys",
        expected_columns=[
            "alloy_id",
            "elements",
            "processing_route",
            "phase",
            "yield_strength_MPa",
            "hardness_HV",
            "density_g_cm3",
        ],
    ),
    "doped_nanoparticles": DatasetConfig(
        domain="doped_nanoparticles",
        version="nanoparticles-v0.1.0",
        source="https://api.catalysis-hub.org/",
        license="CC-BY-4.0",
        target_dir=RAW_DIR / "doped_nanoparticles",
        expected_columns=[
            "reaction_id",
            "chemical_composition",
            "facet",
            "site",
            "adsorbate",
            "reaction_energy_eV",
            "source",
        ],
    ),
}


def ensure_directories() -> None:
    for cfg in CONFIGS.values():
        cfg.target_dir.mkdir(parents=True, exist_ok=True)


def write_schema_file() -> None:
    schema_entries = {
        cfg.domain: {
            "dataset_version": cfg.version,
            "source": cfg.source,
            "license": cfg.license,
            "expected_columns": cfg.expected_columns,
        }
        for cfg in CONFIGS.values()
    }
    with open(SCHEMA_PATH, "w", encoding="utf-8") as handle:
        yaml.safe_dump(schema_entries, handle, sort_keys=True)


def compute_checksum(path: Path) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            sha.update(chunk)
    return sha.hexdigest()


def load_manifest_rows() -> Tuple[List[Dict[str, str]], List[str]]:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError("Provenance manifest missing; ensure T1.1 scaffold is initialized.")
    with open(MANIFEST_PATH, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        return rows, reader.fieldnames or []


def write_manifest_rows(rows: Iterable[Dict[str, str]], fieldnames: List[str]) -> None:
    with open(MANIFEST_PATH, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def update_manifest(domain: str, checksum: str, record_count: int) -> None:
    rows, fieldnames = load_manifest_rows()
    updated = False
    for row in rows:
        if row["domain"] == domain:
            row["checksum_sha256"] = checksum
            row["expected_records"] = str(record_count)
            row["download_status"] = "downloaded"
            row["ingested_on"] = os.environ.get("INGESTION_DATE", DEFAULT_INGESTION_DATE)
            updated = True
            break
    if not updated:
        raise ValueError(f"Domain '{domain}' not found in manifest; add entry before ingestion.")
    write_manifest_rows(rows, fieldnames)


def clean_target_directory(path: Path) -> None:
    for item in path.iterdir():
        if item.is_file():
            item.unlink()
        elif item.is_dir():
            # Preserve nested directories (e.g., CIF structures) but allow user to manage manually.
            continue


def download_perovskites(cfg: DatasetConfig) -> Optional[Tuple[Path, int]]:
    if MPRester is None:
        print("[ERROR] mp-api package not installed. Install with `pip install mp-api`. Skipping perovskite download.")
        return None

    api_key = os.environ.get("MP_API_KEY")
    if not api_key:
        print("[WARN] MP_API_KEY not set; skipping perovskite download.")
        return None

    with MPRester(api_key=api_key) as mpr:
        docs = mpr.materials.summary.search(
            elements=["O"],
            num_elements=3,
            chunk_size=1000,
            fields=[
                "material_id",
                "formula_pretty",
                "symmetry.spacegroup.symbol",
                "band_gap",
                "formation_energy_per_atom",
                "energy_above_hull",
            ],
        )

    if not docs:
        print("[WARN] Materials Project returned no entries for specified criteria.")
        return None

    normalized_rows: List[Dict[str, object]] = []
    for doc in docs:
        spacegroup = None
        symmetry = getattr(doc, "symmetry", None)
        if symmetry is not None:
            spacegroup = getattr(symmetry, "spacegroup_symbol", None) or getattr(symmetry, "symbol", None)
            if hasattr(symmetry, "spacegroup"):
                sg = getattr(symmetry, "spacegroup")
                if isinstance(sg, dict):
                    spacegroup = sg.get("symbol", spacegroup)
                else:
                    spacegroup = getattr(sg, "symbol", spacegroup)

        normalized_rows.append(
            {
                "material_id": doc.material_id,
                "pretty_formula": doc.formula_pretty,
                "spacegroup": spacegroup,
                "band_gap": doc.band_gap,
                "formation_energy_per_atom": doc.formation_energy_per_atom,
                "e_above_hull": doc.energy_above_hull,
            }
        )

    out_path = cfg.target_dir / "materials_project_perovskites.csv"
    clean_target_directory(cfg.target_dir)
    with open(out_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=normalized_rows[0].keys())
        writer.writeheader()
        writer.writerows(normalized_rows)

    return out_path, len(normalized_rows)


def ensure_kaggle_credentials(username: str, key: str) -> None:
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(parents=True, exist_ok=True)
    cred_path = kaggle_dir / "kaggle.json"
    creds = {"username": username, "key": key}
    with open(cred_path, "w", encoding="utf-8") as handle:
        json.dump(creds, handle)
    cred_path.chmod(stat.S_IRUSR | stat.S_IWUSR)


def download_high_entropy_alloys(cfg: DatasetConfig) -> Optional[Tuple[Path, int]]:
    username = os.environ.get("KAGGLE_USERNAME")
    key = os.environ.get("KAGGLE_KEY")
    if not username or not key:
        print("[WARN] Kaggle credentials not set; skipping HEA download.")
        return None

    slug = os.environ.get("KAGGLE_DATASET_SLUG", "miraclelab/high-entropy-alloys")
    try:
        from kaggle import KaggleApi
    except ImportError:
        print("[ERROR] kaggle package not installed. Install with `pip install kaggle`. Skipping HEA dataset.")
        return None

    ensure_kaggle_credentials(username, key)
    api = KaggleApi()
    api.authenticate()

    clean_target_directory(cfg.target_dir)
    cfg.target_dir.mkdir(parents=True, exist_ok=True)
    api.dataset_download_files(slug, path=str(cfg.target_dir), unzip=True)

    csv_files = sorted(cfg.target_dir.glob("*.csv"))
    if not csv_files:
        print(f"[WARN] Kaggle dataset {slug} did not contain CSV files after extraction.")
        return None
    primary = csv_files[0]
    record_count = sum(1 for _ in open(primary, "r", encoding="utf-8")) - 1
    return primary, record_count


def extract_publication(entry: Dict[str, object]) -> Optional[str]:
    publications = entry.get("publications") or entry.get("references")
    if isinstance(publications, list) and publications:
        first = publications[0]
        if isinstance(first, dict):
            return first.get("url") or first.get("doi") or first.get("title")
        return str(first)
    return entry.get("source") or entry.get("publication") or entry.get("reference")


def normalize_catalysis_entry(entry: Dict[str, object]) -> Dict[str, object]:
    return {
        "reaction_id": entry.get("reactionId")
        or entry.get("id")
        or entry.get("reaction_id"),
        "chemical_composition": entry.get("chemical_composition")
        or entry.get("chemicalComposition"),
        "facet": entry.get("facet"),
        "site": entry.get("site")
        or entry.get("site_type")
        or entry.get("siteType"),
        "adsorbate": entry.get("adsorbate")
        or entry.get("adsorbate_name")
        or entry.get("adsorbateName"),
        "reaction_energy_eV": entry.get("reaction_energy")
        or entry.get("energy")
        or entry.get("reactionEnergy"),
        "source": extract_publication(entry),
    }


def parse_catalysis_response(data: object) -> List[Dict[str, object]]:
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if "results" in data and isinstance(data["results"], list):
            return data["results"]
        if "data" in data and isinstance(data["data"], list):
            return data["data"]
        if "data" in data and isinstance(data["data"], dict):
            # GraphQL style: {"data": {"reactions": {"edges": [{"node": ...}]}}}
            reactions = data["data"].get("reactions") if isinstance(data["data"], dict) else None
            if isinstance(reactions, dict) and isinstance(reactions.get("edges"), list):
                return [edge.get("node", {}) for edge in reactions["edges"]]
    return []


def download_doped_nanoparticles(cfg: DatasetConfig) -> Optional[Tuple[Path, int]]:
    limit = int(os.environ.get("CATALYSIS_HUB_DATASET_LIMIT", "5000"))
    pub_id = os.environ.get("CATALYSIS_HUB_PUB_ID", "MamunHighT2019")

    query = """
    query($limit: Int!, $pubId: String!) {
      reactions(first: $limit, pubId: $pubId) {
        edges {
          node {
            id
            chemicalComposition
            surfaceComposition
            facet
            reactionEnergy
            sites
            publication {
              pubId
              title
              doi
            }
          }
        }
      }
    }
    """

    payload = {"query": query, "variables": {"limit": limit, "pubId": pub_id}}

    try:
        response = requests.post(CATALYSIS_HUB_GRAPHQL, json=payload, timeout=120)
        response.raise_for_status()
    except requests.HTTPError as exc:
        print(f"[ERROR] HTTP error from Catalysis-Hub: {exc}")
        return None

    data = response.json()
    edges = data.get("data", {}).get("reactions", {}).get("edges", [])
    if not edges:
        print(f"[WARN] Catalysis-Hub returned no reactions for pubId {pub_id}.")
        return None

    normalized: List[Dict[str, object]] = []
    for edge in edges:
        node = edge.get("node", {})
        publication = node.get("publication") or {}
        normalized.append(
            {
                "reaction_id": node.get("id"),
                "chemical_composition": node.get("chemicalComposition"),
                "surface_composition": node.get("surfaceComposition"),
                "facet": node.get("facet"),
                "sites": node.get("sites"),
                "reaction_energy_eV": node.get("reactionEnergy"),
                "publication_id": publication.get("pubId"),
                "publication_title": publication.get("title"),
                "publication_doi": publication.get("doi"),
            }
        )

    out_path = cfg.target_dir / "catalysis_hub_single_atom_alloy.csv"
    clean_target_directory(cfg.target_dir)
    with open(out_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=normalized[0].keys())
        writer.writeheader()
        writer.writerows(normalized)

    return out_path, len(normalized)


DOMAIN_DOWNLOADERS = {
    "perovskites": download_perovskites,
    "high_entropy_alloys": download_high_entropy_alloys,
    "doped_nanoparticles": download_doped_nanoparticles,
}


def ingest_domain(domain: str) -> None:
    cfg = CONFIGS[domain]
    downloader = DOMAIN_DOWNLOADERS[domain]
    result = downloader(cfg)
    if result is None:
        print(f"[INFO] Skipping manifest update for {domain}; download incomplete.")
        return
    path, record_count = result
    checksum = compute_checksum(path)
    update_manifest(domain, checksum, record_count)
    print(f"[INFO] Updated manifest for {domain} ({record_count} records, checksum {checksum[:12]}…)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and validate project datasets")
    parser.add_argument(
        "domain",
        nargs="?",
        choices=list(CONFIGS.keys()) + ["all"],
        default="all",
        help="Dataset domain to ingest",
    )
    args = parser.parse_args()

    ensure_directories()
    write_schema_file()

    domains = CONFIGS.keys() if args.domain == "all" else [args.domain]
    for domain in domains:
        try:
            ingest_domain(domain)
        except requests.HTTPError as exc:
            print(f"[ERROR] HTTP error while ingesting {domain}: {exc}")
        except Exception as exc:  # pylint: disable=broad-except
            print(f"[ERROR] Unexpected failure for {domain}: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
