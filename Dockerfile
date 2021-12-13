# The Docker container build file

# base image, feel free to change it
FROM python:3.8-slim

RUN mkdir /home/app

WORKDIR /home/app

COPY main.py model.tflite requirements.txt ./

RUN pip install -r requirements.txt

# What is executed when calling the docker container
ENTRYPOINT [ "python", "main.py" ]