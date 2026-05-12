FROM nvcr.io/nvidia/tensorrt-llm/release:0.21.0

# Install lmms-eval from source (pip version has broken task yamls)
RUN cd /tmp && \
    git clone https://github.com/EvolvingLMMs-Lab/lmms-eval.git && \
    cd lmms-eval && \
    pip install -e .

RUN pip install --no-cache-dir \
    # qwen-vl-utils==0.0.8 \
    datasets \
    matplotlib \
    decord

WORKDIR /workspace
