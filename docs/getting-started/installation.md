---
title: Installation
---

# Installation

PyAgenity requires Python 3.10+.

## Using pip

```bash
pip install pyagenity
```

## Optional extras

- Redis-based checkpointing:

```bash
pip install "pyagenity[redis]"
```

- Kafka publisher:

```bash
pip install "pyagenity[kafka]"
```

- RabbitMQ publisher:

```bash
pip install "pyagenity[rabbitmq]"
```

## Dev setup

Clone the repo and install dev tools:

```bash
pip install -r requirements-dev.txt
pip install -e .
```
