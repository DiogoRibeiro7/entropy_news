# syntax=docker/dockerfile:1
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \ 
    && apt-get install --no-install-recommends -y build-essential git \ 
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml poetry.lock ./
RUN pip install --no-cache-dir poetry==1.7.1 \
    && poetry config virtualenvs.create false \
    && poetry install --only main --no-root

COPY entropy_news entropy_news
COPY README.md README.md
COPY docs docs

# Provide an entrypoint for inference-ready workflows
ENTRYPOINT ["python", "-m", "entropy_news.main_forecast"]
