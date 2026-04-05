FROM python:3.10-slim

WORKDIR /app

# TensorFlow ke liye zaroori libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY . /app

# Requirements install karo (Make sure isme 'fastapi' aur 'uvicorn' likha ho)
RUN pip install --no-cache-dir -r requirements.txt

# Render hamesha $PORT variable deta hai, hum 8000 use karenge
EXPOSE 8000

# SABSE IMPORTANT: FastAPI ko run karne ka sahi tarika
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
