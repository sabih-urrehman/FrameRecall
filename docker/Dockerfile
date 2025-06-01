
FROM

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    ffmpeg \
    libzbar0 \
    libzbar-dev \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1
ENV PYTHONOPTIMIZE=2

COPY requirements-compute.txt /tmp/
RUN pip3 install
    pip3 install

WORKDIR /data

CMD ["python3", "--version"]