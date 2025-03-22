# Use the official Python image from the Docker Hub
FROM python:3.13-slim

# Set the working directory in the container
WORKDIR /app

# Install Poetry
RUN pip install --no-cache-dir poetry

# Copy all the files from the current directory to the container
COPY . .

# Install the dependencies
RUN poetry install

# Expose the port the app runs on
EXPOSE 8080

# Command to run the FastAPI server
CMD ["poetry", "run", "start"]
