FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .

# Cloud Run injects $PORT and it is not always 8080. Shell form so it expands.
CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}
