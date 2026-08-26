FROM python:3.11-slim

WORKDIR /app

# Install lightweight system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency manifest and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Expose default Gradio Studio port
EXPOSE 7860

ENV PYTHONUNBUFFERED=1

CMD ["python", "app.py"]