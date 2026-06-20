FROM python:3.11-slim

# libsndfile is required by fairseq2 (SONAR's dependency).
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir ".[api,sonar]"

EXPOSE 8000

CMD ["python", "-m", "prism.api"]
