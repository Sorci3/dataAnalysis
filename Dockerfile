# Runtime image: slim Python with LightGBM runtime deps
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System dependencies required by LightGBM wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps that match the exported MLflow model
COPY model/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy the exported MLflow model (MLmodel, model.pkl, env files)
COPY model/ /app/model/

# Port exposed by `mlflow models serve`
EXPOSE 1234

# Start the model server
CMD ["mlflow", "models", "serve", "-m", "/app/model", "-p", "1234", "-h", "0.0.0.0", "--env-manager", "local"]


