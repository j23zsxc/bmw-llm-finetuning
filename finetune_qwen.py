# -*- coding: utf-8 -*-
import os
import argparse
import torch
import boto3
import logging
from botocore.exceptions import NoCredentialsError, PartialCredentialsError, ClientError
from unsloth import FastLanguageModel
from datasets import load_dataset
from transformers import TrainingArguments, TextStreamer
from trl import SFTTrainer
from unsloth import is_bfloat16_supported

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 定义函数 ---

def formatting_prompts_func(examples):
    """
    格式化数据集以适应 Alpaca 模板。
    使用 'Question' 作为 instruction，'Response' 作为 output。
    Input 字段留空。
    """
    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""
    # 注意：确保这里的键名 ('Question', 'Response') 与你的数据集列名完全匹配（大小写敏感）
    questions = examples["Question"]
    responses = examples["Response"]
    eos_token = "</s>" # 使用 Unsloth/Qwen 可能默认使用的 EOS token，或根据 tokenizer 调整

    texts = []
    for question_text, response_text in zip(questions, responses):
        text = alpaca_prompt.format(
            question_text, # Instruction
            "",            # Input (留空)
            response_text  # Response
        ) + eos_token      # 必须添加 EOS token
        texts.append(text)
    return { "text" : texts }

def upload_directory_to_s3(local_directory, bucket, s3_prefix):
    """将本地目录内容上传到 S3 指定前缀下"""
    s3_client = boto3.client('s3')
    try:
        for root, dirs, files in os.walk(local_directory):
            for filename in files:
                local_path = os.path.join(root, filename)
                # 创建相对路径以保持 S3 上的目录结构
                relative_path = os.path.relpath(local_path, local_directory)
                s3_path = os.path.join(s3_prefix, relative_path).replace("\\", "/") # 确保 S3 路径使用 /

                logger.info(f"正在上传 {local_path} 到 s3://{bucket}/{s3_path}")
                s3_client.upload_file(local_path, bucket, s3_path)
        logger.info(f"成功上传目录 {local_directory} 到 s3://{bucket}/{s3_prefix}")
        return True
    except FileNotFoundError:
        logger.error(f"本地目录 {local_directory} 未找到。")
        return False
    except (NoCredentialsError, PartialCredentialsError):
        logger.error("AWS 凭证未找到或不完整。请确保环境已正确配置 (例如，通过 IRSA 或环境变量)。")
        return False
    except ClientError as e:
        logger.error(f"上传到 S3 时出错: {e}")
        return False
    except Exception as e:
        logger.error(f"上传过程中发生未知错误: {e}")
        return False

# --- 主逻辑 ---

def main(s3_bucket_path=None):
    # --- 模型和分词器设置 ---
    max_seq_length = 2048
    dtype = None # 自动检测
    load_in_4bit = True # 使用 4bit 量化

    logger.info("正在加载模型和分词器...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/Qwen2.5-7B", # 指定模型
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
        # token = "hf_...", # 如果需要访问私有模型，请取消注释并提供 token
    )
    logger.info("模型和分词器加载完成。")

    logger.info("正在添加 LoRA 适配器...")
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
        use_rslora = False,
        loftq_config = None,
    )
    logger.info("LoRA 适配器添加完成。")

    # --- 数据准备 ---
    logger.info("正在加载和处理数据集 'Conard/fortune-telling'...")
    try:
        dataset = load_dataset("Conard/fortune-telling", split = "train")
        # 验证数据集结构
        logger.info(f"数据集特征: {dataset.features}")
        if dataset:
            logger.info(f"第一个样本的键: {list(dataset[0].keys())}")
        else:
            logger.warning("数据集为空！")
            return # 如果数据集为空，则退出

        dataset = dataset.map(formatting_prompts_func, batched = True,)
        logger.info("数据集处理完成。")
        logger.info("格式化后的示例文本:")
        logger.info(dataset[0]["text"][:500] + "...") # 打印部分示例文本
    except Exception as e:
        logger.error(f"加载或处理数据集时出错: {e}")
        return # 发生错误时退出

    # --- 训练 ---
    logger.info("开始训练...")
    output_dir = "outputs" # 本地训练输出目录
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        dataset_num_proc = 2, # 使用多个进程加速数据处理
        packing = False,
        args = TrainingArguments(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4, # 有效 batch size = 2 * 4 = 8
            warmup_steps = 10,               # 增加预热步骤
            max_steps = 100,                 # 增加训练步数以获得更好的效果，根据需要调整
            # num_train_epochs = 1,          # 或者设置训练轮数
            learning_rate = 2e-4,
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            logging_steps = 1,              # 每步都记录日志，以便在 K8s 中查看进度
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = output_dir,        # 指定本地输出目录
            report_to = "none",             # 在 K8s 环境中通常不需要报告给 wandb/tensorboard
        ),
    )

    # 显示初始 GPU 内存状态
    try:
        gpu_stats = torch.cuda.get_device_properties(0)
        start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        logger.info(f"GPU = {gpu_stats.name}. 总显存 = {max_memory} GB.")
        logger.info(f"初始预留显存 = {start_gpu_memory} GB.")
    except Exception as e:
        logger.warning(f"无法获取 GPU 状态: {e}")


    # 开始训练
    trainer_stats = trainer.train()

    # 显示训练后的内存和时间统计
    try:
        used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
        used_percentage = round(used_memory / max_memory * 100, 3) if max_memory > 0 else 0
        lora_percentage = round(used_memory_for_lora / max_memory * 100, 3) if max_memory > 0 else 0
        logger.info(f"训练耗时: {trainer_stats.metrics['train_runtime']:.2f} 秒 ({trainer_stats.metrics['train_runtime']/60:.2f} 分钟).")
        logger.info(f"峰值预留显存 = {used_memory} GB.")
        logger.info(f"训练峰值预留显存 = {used_memory_for_lora} GB.")
        logger.info(f"峰值预留显存占总显存百分比 = {used_percentage} %.")
        logger.info(f"训练峰值预留显存占总显存百分比 = {lora_percentage} %.")
    except Exception as e:
        logger.warning(f"无法计算最终内存统计: {e}")

    logger.info("训练完成。")

    # --- 推理测试 ---
    logger.info("开始推理测试...")
    FastLanguageModel.for_inference(model) # 优化推理速度

    # 使用 Alpaca 模板进行推理
    alpaca_prompt_inference = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

    # 测试用例 1
    instruction_test_1 = "命盤中破軍星在子女宮，對生育有影響嗎"
    inputs_1 = tokenizer(
        [
            alpaca_prompt_inference.format(instruction_test_1, "", "")
        ], return_tensors = "pt").to("cuda")

    logger.info(f"\n--- 推理测试 1 (指令: {instruction_test_1}) ---")
    text_streamer = TextStreamer(tokenizer, skip_prompt=True)
    _ = model.generate(**inputs_1, streamer = text_streamer, max_new_tokens = 128, use_cache = True)
    print("\n(流式输出结束)")

    # 测试用例 2
    instruction_test_2 = "最近事业不顺，想请大师看看我的八字运势。我是1985年12月25日凌晨2点出生的?"
    inputs_2 = tokenizer(
        [
            alpaca_prompt_inference.format(instruction_test_2, "", "")
        ], return_tensors = "pt").to("cuda")

    logger.info(f"\n--- 推理测试 2 (指令: {instruction_test_2}) ---")
    text_streamer_2 = TextStreamer(tokenizer, skip_prompt=True)
    _ = model.generate(**inputs_2, streamer = text_streamer_2, max_new_tokens = 512, use_cache = True) # 增加 token 数量以应对可能更长的回答
    print("\n(流式输出结束)")

    logger.info("推理测试完成。")


    # --- 保存模型 ---
    local_save_path = "lora_model" # 本地保存 LoRA 适配器的目录名

    # 询问用户是否保存模型
    # 在 K8s Job 中，标准输入可能不可用，默认行为可以是不保存，或通过参数控制
    # 这里我们假设可以通过日志判断，或默认执行保存到本地，S3上传由参数控制
    logger.info(f"训练后的 LoRA 适配器将保存在本地目录: {local_save_path}")
    try:
        model.save_pretrained(local_save_path)
        tokenizer.save_pretrained(local_save_path)
        logger.info(f"模型和分词器已成功保存到本地 {local_save_path}")

        # 如果提供了 S3 路径，则尝试上传
        if s3_bucket_path:
            logger.info(f"检测到 S3 路径: {s3_bucket_path}，将尝试上传。")
            # 解析 S3 路径
            if s3_bucket_path.startswith("s3://"):
                s3_parts = s3_bucket_path[5:].split('/', 1)
                if len(s3_parts) == 2:
                    bucket_name = s3_parts[0]
                    s3_prefix = s3_parts[1]
                    logger.info(f"准备上传到 S3 Bucket: {bucket_name}, 前缀: {s3_prefix}")
                    upload_success = upload_directory_to_s3(local_save_path, bucket_name, s3_prefix)
                    if upload_success:
                        logger.info("模型成功上传到 S3。")
                    else:
                        logger.error("模型上传到 S3 失败。")
                else: # 只有 bucket name，没有 prefix
                    bucket_name = s3_parts[0]
                    s3_prefix = "" # 上传到 bucket 根目录
                    logger.info(f"准备上传到 S3 Bucket: {bucket_name} (根目录)")
                    upload_success = upload_directory_to_s3(local_save_path, bucket_name, s3_prefix)
                    if upload_success:
                        logger.info("模型成功上传到 S3。")
                    else:
                        logger.error("模型上传到 S3 失败。")
            else:
                logger.error("提供的 S3 路径格式无效，应为 s3://bucket-name/path/to/save")
        else:
            logger.info("未提供 S3 路径，跳过上传步骤。模型仅保存在本地。")

    except Exception as e:
        logger.error(f"保存模型到本地或上传到 S3 时出错: {e}")

    logger.info("脚本执行完毕。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用 Unsloth 微调 Qwen2.5 7B 模型并可选上传到 S3")
    parser.add_argument(
        "--save-to-s3",
        type=str,
        default=None, # 默认不上传
        help="指定 S3 存储桶和路径以保存微调后的 LoRA 模型，格式：s3://your-bucket-name/your/prefix/"
    )
    args = parser.parse_args()

    main(s3_bucket_path=args.save_to_s3)