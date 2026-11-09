# Use Python 3.11
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy Python source code into the container
COPY src/ ./src/

# Run the Python script
CMD ["python", "src/app/main.py"]