## Use a lightweight Python image
FROM python:slim

## Set the environment variables to prevent Python from writing .pyc files & Ensure Python output is not buffered
## So that output is also not buffered during the creation of the Dockerfile
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

## In case this not works I might need to add the credentials.json to the Dockerfile
## ARG
#ARG GOOGLE_APPLICATION_CREDENTIALS_PATH
#ENV GOOGLE_APPLICATION_CREDENTIALS=/app/credentials.json

## Set the working directory
WORKDIR /app

## Install system dependencies required by LightGBM
## LightGBM requires additional dependencies: libgomp1, clean
## Error: it is not apt-get-install, it is apt-get install
## RUN apt-get update && apt-get-install -y --no-install-recommends \
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

## Copy all the application code from the project directory
COPY . .


## Copy Credentials File
## Way 01 to see which one works
#COPY ${GOOGLE_APPLICATION_CREDENTIALS_PATH}/app/credentials.json
## Way 02 to see which one works
#COPY ${GOOGLE_APPLICATION_CREDENTIALS_PATH} /app/credentials.json

## Install the package in editable mode
## We do not want these type of files __pycache__
## We install the dependencies during the code
## pip install -e . (This is the same but avoiding creating the cache files)
RUN pip install --no-cache-dir -e .

## Train the model before running the application
RUN python pipeline/training_pipeline.py

## Expose the port that Flask will run on
## For this case we expose port 5000 because Flask by default runs on port 5000
EXPOSE 5000

## Command to run the app
#CMD ["python", "application.py"]
CMD ["python", "app.py"]