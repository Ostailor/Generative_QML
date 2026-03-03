FROM python:3.11-slim

WORKDIR /workspace

COPY requirements.txt /workspace/requirements.txt
COPY docs/reproducibility/requirements-lock.txt /workspace/requirements-lock.txt

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /workspace/requirements-lock.txt

COPY . /workspace

CMD ["bash", "-lc", "python scripts/repro/run_repro_check.py --tracking-uri file:///workspace/mlruns"]
