FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml README.md ./
COPY src ./src
COPY alembic ./alembic
COPY alembic.ini ./
COPY scripts/docker-entrypoint.sh /entrypoint.sh

RUN pip install --no-cache-dir -r requirements.txt && pip install --no-cache-dir -e . \
    && chmod +x /entrypoint.sh

RUN mkdir -p /app/data/chroma /app/data/uploads

EXPOSE 8000

ENTRYPOINT ["/entrypoint.sh"]
