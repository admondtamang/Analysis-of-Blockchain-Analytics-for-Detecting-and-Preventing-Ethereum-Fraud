# Use a slim Python base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy only requirements first to leverage Docker layer caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

# Expose the port your app uses (assuming Flask on 5000)
EXPOSE 5000

# Start the app
CMD ["python", "server.py"]
