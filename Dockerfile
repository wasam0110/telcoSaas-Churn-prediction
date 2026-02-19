FROM python:3.10-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements (prefer pinned)
COPY requirements-pinned.txt requirements-pinned.txt
RUN python -m pip install --upgrade pip && pip install -r requirements-pinned.txt

# Copy project
COPY . .

ENV PYTHONUNBUFFERED=1

EXPOSE 8000 8501

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
