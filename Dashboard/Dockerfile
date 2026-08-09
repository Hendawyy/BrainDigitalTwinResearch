FROM python:3.10

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc g++ make \
    libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip

RUN pip install --no-cache-dir \
    "numpy<2.0" \
    torch==2.0.1+cpu \
    torchvision==0.15.2+cpu \
    --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

ENV PYTHONUNBUFFERED=1

ENV OMP_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV PYTHONFAULTHANDLER=1

CMD ["streamlit", "run", "app.py", \
     "--server.port", "8000", \
     "--server.headless", "true", \
     "--server.address", "0.0.0.0"]
