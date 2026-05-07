FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime

# Install system dependencies including C compiler for hdbscan
RUN apt-get update && apt-get install -y \
    git \
    gcc \
    g++ \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

COPY . .

CMD ["python", "main_cluster.py"]