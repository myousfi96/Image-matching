# Use the full Triton image as a base
FROM nvcr.io/nvidia/tritonserver:23.12-py3

# Set the working directory
WORKDIR /workspace

# Copy and install the Python dependencies for quantization
# This layer will be cached by Docker and only re-run if the file changes.
COPY requirements.txt .
RUN pip install -r requirements.txt