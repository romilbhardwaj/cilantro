FROM python:3.7.11-slim-buster

# Set up the development environment
WORKDIR /cilantro

# Install extra packages
RUN apt update && \
    apt install nano

# Install dependencies and numpy (for dragonfly)
COPY requirements.txt setup.py ./
RUN pip install -U pip && \
    pip install --no-cache-dir numpy==1.21.2 && \
    pip install --no-cache-dir -r requirements.txt

# Expose ports for Cilantro Scheduler (grpc)
ENV PORT 10000
EXPOSE $PORT

# Copy the cilantro files
COPY cilantro_clients ./cilantro_clients/
COPY cilantro ./cilantro/
COPY experiments ./demos/

# Install cilantro
RUN pip install -e .

ENV PYTHONUNBUFFERED 1

CMD ["python", "/cilantro/cilantro/driver/incluster_driver.py"]
