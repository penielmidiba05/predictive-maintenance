from fastapi import FastAPI

app = FastAPI(title="Predictive Maintenance API", version="0.2.0")


@app.get("/health")
def health() -> dict:
    """Indique que le service tourne."""
    return {"status": "ok"}
