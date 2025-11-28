<p align="center">
  <img width="700" alt="Image" src="https://github.com/user-attachments/assets/694551bb-f351-4784-a969-7aa86404e248" />
</p>

# What is SeahorseDB?

Seahorse DB is a high-performance vector database designed for fast and efficient similarity searches in large-scale datasets. Whether you're dealing with embeddings from machine learning models or other vectorized data, Seahorse DB provides a scalable solution to store, search, and retrieve vectors with minimal latency.

# SeahorseDB Setup Guide (Docker)

This guide provides instructions to run SeahorseDB using Docker.

---

## Prerequisites

Make sure you have Docker and Docker Compose installed on your system.

- [Install Docker](https://docs.docker.com/get-docker/)
- [Install Docker Compose](https://docs.docker.com/compose/install/)

---

## Build Docker Image

First, clone the repository and initialize submodules:

```bash
git clone https://github.com/SeahorseDB/SeahorseDB.git
cd SeahorseDB
git submodule update --init --recursive
```

Then build the Docker image from the `db_engine` directory:

```bash
docker build -t seahorse:opensource db_engine/
```

---

## How to Run SeahorseDB

### Start with Docker Compose

```bash
docker-compose up
```

This will start SeahorseDB on port `5555`.

### Stop

```bash
docker-compose down
```

### Configuration

You can customize SeahorseDB behavior by editing `seahorsedb.conf`:

```ini
appendonly yes
save ''
port 5555
loglevel notice
bg-read-command-thread 16
enable-in-filter yes
```

The configuration file is mounted into the container via `docker-compose.yaml`.
