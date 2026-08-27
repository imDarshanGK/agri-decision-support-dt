# Dockerfile — builds the Crop Recommendation API + frontend for deployment
# Works on Render, Hugging Face Spaces, Railway, Fly.io, or any Docker host.

FROM python:3.11-slim

WORKDIR /app

# System deps for scikit-learn / matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Render/Railway/HF Spaces inject $PORT — default to 8000 for local docker run
ENV PORT=8000
EXPOSE 8000

# Shell form so $PORT expands at container start
CMD uvicorn api:app --host 0.0.0.0 --port $PORT
