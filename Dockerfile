# Use an official lightweight Python image.
# https://hub.docker.com/_/python
FROM python:3.10-slim-buster

# Set the working directory
WORKDIR /eetc-strategy-runner-one-time

# Copy local code to the container image.
COPY . .

# Install dependencies
RUN pip3 install -r requirements.txt

# Run the web service on container startup. Here we use the uvicorn webserver
# For environments with multiple CPU cores, increase the number of workers to
# be equal to the cores available.
CMD uvicorn main:app --host 0.0.0.0 --port $PORT
