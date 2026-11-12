# FetalAI backend (CPU inference). Build from the repository root:
#   docker compose build backend
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FETALAI_CHECKPOINT_PATH=/models/hc18_unet_best.pth \
    FETALAI_PLANE_CHECKPOINT_PATH=/models/plane_classifier_best.pth \
    FETALAI_DEVICE=cpu

WORKDIR /app

COPY requirements.txt .
# Recent pip resumes interrupted downloads (large torch wheel on slow or flaky connections).
RUN pip install --upgrade "pip>=25.1" \
 && pip install --resume-retries 10 --timeout 120 \
      torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --resume-retries 10 --timeout 120 -r requirements.txt

COPY ml ml
COPY backend backend
COPY outputs/test_metrics.json outputs/test_metrics.json
COPY outputs/plane_metrics.json outputs/plane_metrics.json

RUN useradd --create-home --uid 10001 app
USER app

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=5 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health', timeout=4)"

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
