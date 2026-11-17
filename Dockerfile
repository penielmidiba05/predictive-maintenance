FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Bibliothèque système requise par LightGBM
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Dépendances d'abord (couche mise en cache)
COPY requirements-api.txt .
RUN pip install -r requirements-api.txt

# Puis le code et le modèle
COPY src/ src/
COPY app/ app/
COPY configs/ configs/
COPY models/ models/

# Ne pas tourner en administrateur
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
