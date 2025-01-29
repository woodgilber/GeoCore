#FROM ubuntu:jammy-20221101
FROM python:3.10.12
#FROM arm64v8/ubuntu:jammy-20221101
LABEL maintainer="Zanskar ML Team"
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED True

RUN apt-get update && apt-get -y install \
    apt-transport-https \ 
    build-essential \
    ca-certificates \
    curl \
    gnupg \
    libgdal-dev \
    vim 

# Install CLI
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.asc] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | tee /usr/share/keyrings/cloud.google.asc
RUN apt-get update && apt-get -y install google-cloud-cli

RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:$PATH"

RUN mkdir /modeling
COPY configs /modeling/configs
COPY MLGEOCORE_PUBLIC /modeling/modeling/
COPY apps /modeling/apps/

COPY poetry.lock pyproject.toml /modeling/

COPY README.md /modeling

WORKDIR /modeling

# Project initialization:
RUN poetry config virtualenvs.create false \
  && poetry install

RUN sed -i 's/np.int/int/g'  /usr/local/lib/python3.10/site-packages/skopt/space/transformers.py