"""Minimal FastAPI entrypoint — extend with prediction routes."""

from fastapi import FastAPI

app = FastAPI(
    title="Resilient Cocoa Model API",
    description="Geospatial ML inference service",
    version="0.1.0",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
