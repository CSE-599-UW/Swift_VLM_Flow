# Source this before building engines or running TensorRT-Edge-LLM binaries on the GB10.
#   source edge-deployment/benchmark/speculative/env.sh
# Set up during the EAGLE3 SD benchmark (2026-05-31). GB10 = aarch64, sm_121, CUDA 13.0.

export CUDA_HOME=/usr/local/cuda-13.0
export PATH=$CUDA_HOME/bin:$PATH

# Locally-extracted TensorRT 10.13.3.9 (no system install)
export TRT_PACKAGE_DIR=$HOME/trt-10.13.3.9/usr

# TensorRT-Edge-LLM checkout (built for sm_121)
export EDGE_LLM_PATH=$HOME/TensorRT-Edge-LLM

# Runtime needs TRT + CUDA libs on the loader path
export LD_LIBRARY_PATH=$TRT_PACKAGE_DIR/lib/aarch64-linux-gnu:$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Workspace holding models/onnx/engines for the benchmark
export WORKSPACE_DIR=$HOME/tensorrt-edgellm-workspace
export MODEL_NAME=Qwen2.5-VL-7B-Instruct
