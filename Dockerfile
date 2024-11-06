# Use the official Selenium Standalone Chrome image as the base image
FROM python:3.12

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file to the container and install dependencies
COPY requirements.txt /app/requirements.txt

# Switch to root user to install packages and upgrade pip
USER 0

# Upgrade pip to the latest version
RUN pip install --upgrade pip

# Install the dependencies from requirements.txt
RUN pip3 install -r requirements.txt

# Copy your FastAPI Python script to the container
COPY . .

# Expose the port that FastAPI will run on
EXPOSE 4050

# Set the command to run your FastAPI application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "4050"]
