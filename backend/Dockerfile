# Use official Python image from DockerHub
FROM python:3.11-slim

# Install ffmpeg (needed for pydub)
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy all backend files into the container
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Tell Flask where models are
ENV FLUENTA_MODELS_DIR=/app/models_retrained


# Expose the port Flask runs on
EXPOSE 5000

# Run the app using Gunicorn (production WSGI server)
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
