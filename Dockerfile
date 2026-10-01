FROM python:3.11-slim

# System libs needed by matplotlib (fontconfig) and h5py's HDF5 backend.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libhdf5-dev \
    fontconfig \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY . .
RUN pip install --no-cache-dir -e .

CMD ["bash"]
