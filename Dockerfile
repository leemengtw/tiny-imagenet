# Base image
FROM python:3.6
MAINTAINER Meng Lee "b98705001@gmail.com"

# Updating repository sources
RUN apt-get update

# Set working space
WORKDIR /tiny-imagenet

# Copying requirements.txt file and all other relative content
COPY ./requirements.txt /tiny-imagenet/requirements.txt
COPY ./Dockerfile /tiny-imagenet/Dockerfile
COPY ./NaiveResNet.py /tiny-imagenet/NaiveResNet.py
COPY ./tiny-imagenet.ipynb /tiny-imagenet/tiny-imagenet.ipynb
COPY ./tinyimagenet.py /tiny-imagenet/tinyimagenet.py

# pip install
RUN pip install --no-cache -r requirements.txt

# Exposing ports
EXPOSE 8888

# Running jupyter notebook
# --NotebookApp.token ='demo' is the password
CMD ["jupyter", "notebook", "--no-browser", "--ip=0.0.0.0", "--allow-root", "--NotebookApp.token='demo'"]