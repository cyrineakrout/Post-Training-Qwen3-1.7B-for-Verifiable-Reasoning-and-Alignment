# CS-552 standard-project image template
#
# Most teams should use the default course image directly and do not need
# this Dockerfile. It is provided as a starting point only if your project
# genuinely needs a custom image with extra system packages or libraries.
#
# Built on the NVIDIA CUDA 12.8 base image, with PyTorch 2.10.0, vLLM
# 0.19.1, and FlashAttention 2.8.3 installed from prebuilt wheels (no
# source builds, no cluster-specific mounts). We add the Python NLP
# stack (transformers, TRL, PEFT, datasets, JupyterLab, eval/RAG libs)
# on top.

FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

SHELL ["/bin/bash", "-cu"]

ARG python=3.12
ENV PYTHON_VERSION=${python}

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_CONSTRAINT=/tmp/torch-constraints.txt \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false

# CUDA & build environment. The wheels below are prebuilt, so these
# matter only if pip falls back to a source build. Kept for parity with
# the upstream pin set.
ENV TORCH_CUDA_ARCH_LIST='9.0+PTX' \
    VLLM_FA_CMAKE_GPU_ARCHES='90-real' \
    FORCE_CUDA=1 \
    MAX_JOBS=64 \
    NVCC_THREADS=8 \
    USE_CUDNN=1 \
    VLLM_FLASH_ATTN_VERSION=3 \
    CMAKE_BUILD_PARALLEL_LEVEL=64 \
    FLASHINFER_ENABLE_AOT=1 \
    CMAKE_POLICY_VERSION_MINIMUM=3.5 \
    NCCL_IB_DISABLE=1 \
    NCCL_NET=Socket

ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

RUN apt-get update && apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends \
        build-essential cmake \
        git git-lfs vim less tmux htop nano screen \
        curl wget unzip 7zip \
        ca-certificates apt-utils gnupg sudo openssh-client \
        librdmacm1 libibverbs1 ibverbs-providers \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-dev python-is-python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

# System libs commonly needed for vision / multimedia data pipelines.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgtk2.0-dev pkg-config \
        libavcodec-dev libavfilter-dev libavformat-dev libavutil-dev libswscale-dev \
        libtbbmalloc2 libtbb-dev \
        libjpeg-dev libpng-dev libtiff-dev \
        ffmpeg libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 \
        ccache gcc-12 g++-12 \
        libtcmalloc-minimal4 libnuma-dev jq lsof \
    && rm -rf /var/lib/apt/lists/*

# Pin torch + vllm via a constraints file so later `pip install`s never
# silently upgrade them.
RUN printf '%s\n' \
        'torch==2.10.0+cu128' \
        'torchvision==0.25.0+cu128' \
        'torchaudio==2.10.0+cu128' \
        'vllm==0.19.1' \
        > "${PIP_CONSTRAINT}"

RUN python -m pip install --ignore-installed \
        "packaging>=24.2" \
        "setuptools>=77.0.3,<81.0.0" \
        wheel ninja pybind11

# Torch (pinned, CUDA 12.8 build).
RUN python -m pip install \
        "torch==2.10.0+cu128" \
        "torchvision==0.25.0+cu128" \
        "torchaudio==2.10.0+cu128" \
        --index-url https://download.pytorch.org/whl/cu128 && \
    python -c "import torch, torchvision, torchaudio; \
        assert torch.__version__ == '2.10.0+cu128', torch.__version__; \
        assert torch.version.cuda == '12.8', torch.version.cuda; \
        assert torchvision.__version__ == '0.25.0+cu128', torchvision.__version__; \
        assert torchaudio.__version__ == '2.10.0+cu128', torchaudio.__version__"

# FlashAttention 2.8.3 (prebuilt wheel matching torch 2.10 + cu128).
RUN python -m pip install \
        https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.0/flash_attn-2.8.3%2Bcu128torch2.10-cp312-cp312-linux_x86_64.whl

# vLLM 0.19.1 (prebuilt wheel).
RUN python -m pip install --extra-index-url https://download.pytorch.org/whl/cu128 \
        https://github.com/vllm-project/vllm/releases/download/v0.19.1/vllm-0.19.1-cp38-abi3-manylinux_2_31_x86_64.whl && \
    python -c "import vllm; assert vllm.__version__.split('+')[0] == '0.19.1', vllm.__version__"

# Standard-project Python requirements. The PIP_CONSTRAINT file above
# keeps torch/vllm pinned no matter what these packages ask for.
COPY requirement_docker.txt /tmp/requirement_docker.txt
RUN python -m pip install -r /tmp/requirement_docker.txt

# Course tooling on top of the requirements file:
#   - jupyterlab + ipykernel/ipywidgets: required by `submit.sh`, which
#     launches `jupyter lab` inside the pod.
#   - hf_transfer: paired with HF_HUB_ENABLE_HF_TRANSFER=1 (set above)
#     for ~10x faster Hugging Face downloads.
RUN python -m pip install \
        "jupyterlab>=4.2" "ipykernel" "ipywidgets" "jupyterlab-widgets" \
        "hf_transfer"

# Final sanity check: all the pinned bits still import and report the
# expected versions after every layer above.
RUN python -c "import torch, torchvision, torchaudio, vllm, transformers, trl, peft, datasets; \
    assert torch.__version__ == '2.10.0+cu128', torch.__version__; \
    assert torch.version.cuda == '12.8', torch.version.cuda; \
    assert vllm.__version__.split('+')[0] == '0.19.1', vllm.__version__; \
    print('torch', torch.__version__, 'cuda', torch.version.cuda); \
    print('vllm', vllm.__version__); \
    print('transformers', transformers.__version__)"

WORKDIR /workspace
CMD ["/bin/bash"]
