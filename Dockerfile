FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY Makefile /app/
COPY requirements.txt /app/
COPY run.py /app/
COPY constants.yaml /app/
COPY . /app/

RUN apt-get update && apt-get install -y make libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt

RUN ls -l /app

CMD ["make", "run-pipeline"]