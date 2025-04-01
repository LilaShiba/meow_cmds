# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies (for PyTorch and other libraries)
RUN apt-get update && apt-get install -y \
    build-essential \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy the current directory contents into the container
COPY . .

# Install any needed Python packages from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 5000 for backend API (optional)
EXPOSE 5000

# Run the training script as the container starts
CMD ["python3", "utils/distributed_trainer.py"]
