FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8000 \
    GATEWAY_DATABASE_URL=sqlite:////data/gateway.db

COPY pyproject.toml README.md ./
COPY src ./src
COPY benchmarks ./benchmarks
RUN pip install --no-cache-dir .
RUN useradd --create-home --uid 10001 gateway \
    && mkdir -p /data \
    && chown gateway:gateway /data
USER gateway
EXPOSE 8000
CMD ["sh", "-c", "uvicorn gateway.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
