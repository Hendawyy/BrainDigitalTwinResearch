FROM python:3.10

WORKDIR /app

# System deps for MONAI/pydicom, plus a compiler toolchain some of the
# scientific packages need to build native extensions.
RUN apt-get update && apt-get install -y \
    gcc g++ make \
    libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip

# torch==2.0.1+cpu / torchvision==0.15.2+cpu only exist on PyTorch's own CPU
# wheel index, not on regular PyPI — install those first from there, then the
# rest of requirements.txt from PyPI as usual.
RUN pip install --no-cache-dir \
    "numpy<2.0" \
    torch==2.0.1+cpu \
    torchvision==0.15.2+cpu \
    --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

ENV PYTHONUNBUFFERED=1

# The B2 App Service plan has 2 vCPUs. torch, numpy and scikit-learn each
# default their BLAS/OpenMP thread pool to the host's full core count, which on
# a constrained container is a known crash/instability source alongside the
# file-watcher segfault (see .streamlit/config.toml's fileWatcherType). Pin to 1.
ENV OMP_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV MKL_NUM_THREADS=1

# v18/v19 both die with exit code 139 (SIGSEGV) on a fixed ~2-minute interval,
# unrelated to user interaction — a plain Python exception can't explain that,
# and Python can't normally produce a traceback for a native crash at all.
# faulthandler installs a low-level signal handler that dumps the Python-level
# stack (which line was executing) to stderr the instant SIGSEGV/SIGABRT/etc.
# arrive, *before* the process dies — this is the fastest way to find out where
# it's actually crashing instead of guessing again. Captured by
# `az webapp log tail` since PYTHONUNBUFFERED is already set above.
ENV PYTHONFAULTHANDLER=1

CMD ["streamlit", "run", "app.py", \
     "--server.port", "8000", \
     "--server.headless", "true", \
     "--server.address", "0.0.0.0"]
