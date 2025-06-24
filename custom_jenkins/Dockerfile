
## Use the Jenkins image as base image
FROM jenkins/jenkins:lts

## Initially it is User Jenkins which does not have permission to install libraries
## Switch to root user to install dependencies
USER root

## Install prerequisites and Docker
## Updating, installing Docker inside of this, then installing Docker CLI
RUN apt-get update -y && \
    apt-get install -y apt-transport-https ca-certificates curl gnupg software-properties-common && \
    curl -fsSL https://download.docker.com/linux/debian/gpg | apt-key add - && \
    echo "deb [arch=amd64] https://download.docker.com/linux/debian bullseye stable" > /etc/apt/sources.list.d/docker.list && \
    apt-get update -y && \
    apt-get install -y docker-ce docker-ce-cli containerd.io && \
    apt-get clean

## Add Jenkins user to the Docker group (create if it does not exist)
RUN groupadd -f docker && \
    usermod -aG docker jenkins

## Create the Docker directory and volume for (DinD) Docker in Docker
RUN mkdir -p /var/lib/docker
VOLUME /var/lib/docker

## Switch back to Jenkins user
USER jenkins