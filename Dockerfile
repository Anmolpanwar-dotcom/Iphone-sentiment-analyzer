FROM python:3.10-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

# Render ke liye port 8000 use karenge
CMD ["uvicorn", "main.py:app", "--host", "0.0.0.0", "--port", "8000"]