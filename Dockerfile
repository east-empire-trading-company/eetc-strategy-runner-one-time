# Use an official lightweight Python image.
# https://hub.docker.com/_/python
FROM python:3.10-slim-buster

RUN apt-get update && apt-get install -y make gfortran libfreetype6-dev libhdf5-dev liblapack-dev libopenblas-dev libpng-dev

ENV TA_PREFIX="/opt/ta-lib-core"
ENV TA_LIBRARY_PATH="$TA_PREFIX/lib"
ENV TA_INCLUDE_PATH="$TA_PREFIX/include"

# Set the working directory
WORKDIR /eetc-strategy-runner-one-time

# Copy local code to the container image.
COPY . .

RUN cd talib_source/ && ./configure --prefix=$TA_PREFIX && make && make install && cd ..

# Install dependencies
RUN pip3 install -r requirements.txt
RUN cd ibapi_source/ && python3 setup.py bdist_wheel && cd ..
RUN cd ibapi_source/ && pip3 install --upgrade dist/ibapi-10.19.1-py3-none-any.whl && cd ..

# Run the web service on container startup. Here we use the uvicorn webserver
# For environments with multiple CPU cores, increase the number of workers to
# be equal to the cores available.
CMD uvicorn main:app --host 0.0.0.0 --port $PORT
