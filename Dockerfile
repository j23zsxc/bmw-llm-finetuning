# 选择一个包含 CUDA Toolkit 的基础镜像
# 确保 CUDA 版本与 unsloth/torch 兼容，并且与 EKS 节点的驱动兼容
# Unsloth 推荐 CUDA 12.1 或更高版本
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# 设置环境变量，避免交互式安装提示
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC \
    # 设置 Hugging Face Hub 缓存目录（可选，但在容器内有用）
    HF_HOME=/root/.cache/huggingface \
    # 设置 Transformers 缓存目录（可选）
    TRANSFORMERS_CACHE=/root/.cache/huggingface/transformers

# 安装基础工具和 Python
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# 创建工作目录
WORKDIR /app

# 复制 Python 脚本
COPY finetune_qwen.py /app/finetune_qwen.py

# 安装 Python 依赖
# 注意：根据 unsloth 的最新建议调整依赖项和版本
# bitsandbytes 可能需要特定 CUDA 版本的 wheel
# xformers 通常是可选的，但可以加速；版本需要与 PyTorch/CUDA 匹配
RUN pip3 install --no-cache-dir --upgrade pip
RUN pip3 install --no-cache-dir \
    torch==2.3.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121
RUN pip3 install --no-cache-dir \
    "unsloth[cu121-ampere-torch231] @ git+https://github.com/unslothai/unsloth.git" \
    "transformers==4.43.3" \
    "datasets==2.20.0" \
    "trl==0.9.6" \
    "accelerate==0.32.1" \
    "bitsandbytes==0.43.1" \
    "peft==0.12.0" \
    "boto3==1.34.145" \
    "sentencepiece==0.2.0" \
    "protobuf==4.25.3" \
    "huggingface_hub==0.24.1" \
    "hf_transfer==0.1.6" \
    "xformers==0.0.27" # 确保版本与 torch/cuda 兼容

# 授予脚本执行权限（如果需要）
# RUN chmod +x /app/finetune_qwen.py

# 设置默认启动命令
# 参数将在 K8s Job 定义中覆盖或添加
CMD ["python3", "finetune_qwen.py"]