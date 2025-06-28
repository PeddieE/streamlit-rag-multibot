# Use an official Python runtime as a parent image
# FROM python:3.9-slim-buster
FROM python:3.11-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# --no-cache-dir is good for smaller image size
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire application code into the container at /app
COPY . .

# Expose the default Streamlit port (8501)
EXPOSE 8501

# Define the command to run your Streamlit application
# Replace 'your_chatbot_app.py' with the actual name of your main Streamlit file
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]