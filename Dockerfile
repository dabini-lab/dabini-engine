# Use the official Python image from the Docker Hub
FROM python:3.13-slim

# Set the working directory in the container
WORKDIR /app

# Install Poetry
RUN pip install --no-cache-dir poetry

# Copy the pyproject.toml and poetry.lock files into the container
COPY pyproject.toml poetry.lock ./

# Install the dependencies
RUN poetry install --no-root

# Copy the rest of the application code into the container
COPY . .

# Expose the port the app runs on
EXPOSE 8080

# Command to run the FastAPI server
CMD ["poetry", "run", "uvicorn", "dabini_engine.main:api", "--host", "0.0.0.0", "--port", "8080"]
