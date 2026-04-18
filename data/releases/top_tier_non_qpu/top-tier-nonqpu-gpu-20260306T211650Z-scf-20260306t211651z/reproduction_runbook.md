# Reproduction Runbook (Real DFT Campaign v1)

## 1) Environment
1. Activate the project environment.
2. Export `MLFLOW_TRACKING_URI=file:///home/om/Projects/Generative_QML/mlruns`.

## 2) Validate campaign artefacts
Run:
```bash
python scripts/dft/validate_production_outputs.py --campaign-id top-tier-nonqpu-gpu-20260306T211650Z-scf-20260306t211651z
```

## 3) Verify release checksums
Run:
```bash
python - <<'PY'
import hashlib, json
from pathlib import Path
root = Path('data/releases/real_dft_campaign_v1')
manifest = json.loads((root/'release_manifest.json').read_text())
for entry in manifest:
    path = root / entry['path']
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == entry['sha256'], f"checksum mismatch: {entry['path']}"
print('checksum verification passed')
PY
```

## 4) Refresh acceptance tracking
Run:
```bash
python tracking/reporting/update_status_snapshot.py --mlflow-tracking-uri file://$(pwd)/mlruns
```

## 5) Expected headline values
- `label_efficiency_gain`: 0.8
- `valid_candidates`: 60
- `completed_jobs`: 60
