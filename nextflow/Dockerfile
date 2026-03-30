FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /workspace

COPY nextflow/requirements-fe.txt /tmp/requirements-fe.txt
RUN pip install --no-cache-dir -r /tmp/requirements-fe.txt

COPY nextflow /workspace/nextflow

ENTRYPOINT ["python", "--version"]
