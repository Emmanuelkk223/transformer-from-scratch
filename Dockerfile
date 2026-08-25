FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /app

# Set PYTHONPATH so internal modules are discoverable globally
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Run pytest through the python module wrapper
RUN python -m pytest tests/ -v

ENTRYPOINT ["python", "generate.py"]
CMD ["--text", "The quick brown fox jumps over the lazy dog.", "--beam_size", "5"]