# High-Performance YOLOv8 Inference Server

This project implements a production-grade, real-time video inference system using YOLOv8. It's built on a decoupled client-server architecture to achieve maximum throughput, low latency, and optimal resource utilization.

The system is composed of two main components:
* **`server_formatted.py`**: A multi-threaded, GPU-accelerated server that runs the SEDA (Staged Event-Driven Architecture) pipeline for high-performance inference.
* **`client.py`**: A lightweight, `asyncio`-based client that streams video frames (from a file or webcam) to the server and saves the received JSON results to a file.


## Key Features

* **Client-Server Architecture:** Decouples video capture (client) from heavy computation (server).
* **High-Throughput Pipeline:** The server uses a multi-threaded pipeline. Each step (receive, pre-process, infer, post-process) runs in its own thread to prevent I/O from blocking computation.
* **Low-Latency Inference:** Utilizes `onnxruntime-gpu` to run the YOLOv8 model on an NVIDIA GPU.
* **Asynchronous I/O:** Both client and server use `asyncio` and `websockets` for efficient, non-blocking network communication.
* **Structured JSON Output:** Emits inference results in a clean, standardized JSON format, which the client saves to `results.jsonl`.

## Setup and Installation

This project **requires an NVIDIA GPU** on the server machine to achieve its performance goals.

### 1. Project Files

Download all the project files (`server_formatted.py`, `client.py`, `export_model.py`, `requirements.txt`) into a single directory.

### 2. Python Environment

It is highly recommended to use a virtual environment.

```bash
# Using conda
conda create -n yolo_server python=3.10
conda activate yolo_server

# Or using venv
python -m venv yolo_env
source yolo_env/bin/activate # (Linux/macOS)
yolo_env\Scripts\activate     # (Windows)
