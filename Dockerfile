FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    PORT=8081 \
    DB_USER=postgres \
    DB_PASSWORD=admin \
    DB_HOST=localhost \
    DB_PORT=5432 \
    DB_NAME=banco_vazao

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential libpq-dev && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN chmod +x docker-entrypoint.sh

EXPOSE 8081

ENTRYPOINT ["./docker-entrypoint.sh"]
