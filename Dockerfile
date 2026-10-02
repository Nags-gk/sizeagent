FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends ngspice git && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -e .[dev] && ./scripts/setup_pdk.sh
CMD ["pytest", "-q"]
