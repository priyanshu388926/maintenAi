# Multi-stage build: compile the React front end, then run FastAPI with the ML package.
FROM node:20-alpine AS web
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY ml ./ml
COPY backend ./backend
COPY reports ./reports
COPY --from=web /web/dist ./frontend/dist
ENV PYTHONUNBUFFERED=1
EXPOSE 8000
# The dataset is mounted at runtime (data/raw is not baked into the image); models are trained on first start.
CMD ["sh", "-c", "python -c 'from ml.inference.predictor import Predictor; Predictor(\"engineered\"); Predictor(\"raw\")' 2>/dev/null || python -m ml.inference.train; exec uvicorn app.main:app --app-dir backend --host 0.0.0.0 --port 8000"]
