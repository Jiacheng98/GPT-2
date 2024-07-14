# Use the official Python image as the base image
FROM robd003/python3.10:latest

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Run the application
ENTRYPOINT ["python", "main.py"]
