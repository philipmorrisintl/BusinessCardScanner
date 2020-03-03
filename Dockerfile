# We Use an official Python runtime as a parent image
FROM python:3.6.8-stretch

# Some local variables
ARG ROOT_DIR=/contacts-app
ARG STARTUP_SCRIPT=${ROOT_DIR}/start-server.sh

# The enviroment variable ensures that the python output is set straight
# to the terminal with out buffering it first
ENV PYTHONUNBUFFERED 1

# create root directory for our project in the container
RUN mkdir ${ROOT_DIR}

# Set the working directory to ROOT_DIR
WORKDIR ${ROOT_DIR}

# Install any needed packages
RUN pip install Django==2.1.7
RUN pip install django-neomodel==0.0.4
RUN pip install sklearn-crfsuite==0.3.6
RUN pip install django-extensions==2.1.9
RUN pip install xmljson==0.2.0
RUN pip install environs==4.2.0
RUN pip install rasa[spacy]==1.1.4

# download spacy model
RUN python -m spacy download en_core_web_md
RUN python -m spacy link en_core_web_md en

RUN pip install names-dataset

RUN pip install opencv-python
RUN pip install opencv-contrib-python

# App needs to wait for neo4j
RUN apt-get install -y git
RUN git clone https://github.com/vishnubob/wait-for-it.git

RUN pip install scipy==1.3.0
RUN pip install pdf2image==1.9.0

RUN apt-get update && apt-get install -y libpoppler-private-dev libpoppler-cpp-dev
RUN apt-get update && apt-get install poppler-utils

# RUN pip install shapely

# Copy the current directory contents into the container at ROOT_DIR
ADD . ${ROOT_DIR}/

# Database directory
RUN mkdir ${ROOT_DIR}/data

# Startup script
RUN chmod u+x ${STARTUP_SCRIPT}
ENV STARTUP_SCRIPT=${STARTUP_SCRIPT}

# Entrypoint: run app
ENTRYPOINT wait-for-it/wait-for-it.sh neo4j:7687 -t 300 -- \
           wait-for-it/wait-for-it.sh libpostal:8080 -t 300 -- \
           wait-for-it/wait-for-it.sh nlu:8081 -t 300 -- \
           ${STARTUP_SCRIPT}
