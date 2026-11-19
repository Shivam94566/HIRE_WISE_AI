# Official Python 3.12 slim image for minimal footprint
FROM python:3.12-slim

# Set working directory inside container
WORKDIR /app

# Copy dependencies list first for efficient Docker layer caching
COPY requirements.txt .

# Install dependencies without local cache
RUN pip install --no-cache-dir -r requirements.txt

# Copy all MLOps service code into container
COPY . .

# Expose FastAPI listening port
EXPOSE 8000

# Start production server using uvicorn
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]
