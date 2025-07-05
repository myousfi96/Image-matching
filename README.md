# Product Matching System

This repository contains a comprehensive system for finding products through visual and text-based search. It uses deep learning models served via NVIDIA Triton for high-performance inference and a vector database for efficient similarity search. The entire application stack is containerized using Docker and orchestrated with Docker Compose.

## Architecture

The system is composed of four main containerized services:

- **Frontend**: A Streamlit application that provides the user interface for searching products.
- **API**: A FastAPI backend that handles business logic, communicates with Triton and the databases.
- **Triton Server**: The NVIDIA Triton Inference Server, which serves the computer vision (DINOv2) and NLP (BGE) models. It also runs a startup script to populate the databases.
- **MongoDB**: A database for storing product metadata.

## Features
- **Image-based Search**: Upload a product image to find visually similar items.
- **Text-based Search**: Describe a product to find relevant items.
- **High-Performance Inference**: Utilizes NVIDIA Triton Inference Server for fast and scalable model serving.
- **Efficient Search**: Leverages a vector database for quick similarity lookups.
- **RESTful API**: Clean, documented API for easy integration.
- **System Health Monitoring**: Endpoint to check the status of all components.

## Prerequisites

Before you begin, ensure you have the following installed on your system:

1.  **Docker**: [Install Docker](https://docs.docker.com/engine/install/)
2.  **Docker Compose**: [Install Docker Compose](https://docs.docker.com/compose/install/) (usually included with Docker Desktop).
3.  **NVIDIA GPU**: A compatible NVIDIA GPU with the latest drivers.

## Setup & Configuration

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/myousfi96/Image-matching.git
    cd Image-matching
    ```
2.  **Data Population**
    The `populate_db.py` script is automatically run by the `triton-server` container on its first startup. This script populates the vector database and MongoDB with initial product data.
    
    **Sample dataset**: A small set of example product images and a matching `metadata.json` file are already included under the `data/` directory (`data/images/…`). If you do nothing, running `docker-compose up --build` will use these files and the startup script will automatically populate both databases for you.
    
    If you want to experiment with your own catalogue, simply replace the contents of `data/images/` and update `data/metadata.json` before starting the stack.

## Running the System

To build and run all the services, execute the following command from the root of the project:

```bash
docker-compose up --build
```
## Accessing the Services

- **Frontend Application**:
  Open your web browser and navigate to `http://localhost:8501`.

- **API Documentation**:
  The FastAPI backend's documentation is available at:
  - Swagger UI: `http://localhost:8080/docs`
  - ReDoc: `http://localhost:8080/redoc`

## Stopping the System

To stop all running containers, press `Ctrl+C` in the terminal where `docker-compose up` is running, or run the following command from another terminal:

```bash
docker-compose down
```

This will stop and remove the containers. To remove the data volumes as well, use:
```bash
docker-compose down -v
``` 