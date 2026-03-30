FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /workspace

COPY nextflow/requirements-fe.txt /tmp/requirements-fe.txt
RUN pip install --no-cache-dir -r /tmp/requirements-fe.txt
RUN apt-get update && apt-get install -y --no-install-recommends curl unzip \
    && curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "/tmp/awscliv2.zip" \
    && unzip -q /tmp/awscliv2.zip -d /tmp \
    && /tmp/aws/install \
    && rm -rf /var/lib/apt/lists/* /tmp/aws /tmp/awscliv2.zip

COPY nextflow /workspace/nextflow

# Do not set a Python entrypoint here.
# Nextflow AWS Batch injects its own `bash -o pipefail -c ...` command.
