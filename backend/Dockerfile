FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

COPY backend/requirements.txt ./requirements.txt

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

COPY backend ./backend
COPY ml ./ml
COPY models ./models

EXPOSE 8000

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]