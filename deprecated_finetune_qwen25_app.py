# -*- coding: utf-8 -*-
import torch
import os
import boto3 # 用于S3操作
import shutil # 用于删除临时目录
from urllib.parse import urlparse # 用于解析S3路径
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments, TextStreamer
from unsloth import is_bfloat16_supported

# --- 1. 模型和分词器设置 ---
max_seq_length = 2048
dtype = None # 自动检测, Ampere架构 (如 A10G) 支持 bfloat16
load_in_4bit = True # 使用4位量化以节省显存

# 加载模型和分词器
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen2.5-7B", # 使用 Qwen2.5 7B 模型
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# --- 2. LoRA 设置 ---
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # LoRA 秩
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",], # Qwen2.5 的目标模块
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth", # 优化显存使用
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)

# --- 3. 数据准备 ---
# 定义 Alpaca 格式的 Prompt 模板
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

# 定义 EOS token
EOS_TOKEN = tokenizer.eos_token # 获取模型的 EOS token
if not EOS_TOKEN:
    print("警告：无法自动获取 EOS token，手动设置为 '</s>'")
    EOS_TOKEN = "</s>" # 如果无法自动获取，可以手动设置

# 数据集格式化函数
def formatting_prompts_func(examples):
    questions = examples["Question"]
    responses = examples["Response"]
    texts = []
    for question_text, response_text in zip(questions, responses):
        # 将数据集字段映射到 Alpaca 模板
        # Question -> Instruction, Input 留空, Response -> Response
        text = alpaca_prompt.format(
            question_text, # instruction
            "",            # input
            response_text  # response
        ) + EOS_TOKEN # 必须添加 EOS_TOKEN
        texts.append(text)
    return { "text" : texts }

# 加载数据集 (与原始代码一致)
print("正在加载数据集 'Conard/fortune-telling'...")
dataset = load_dataset("Conard/fortune-telling", split = "train")
print("数据集加载完成。")

# 验证数据集结构
print("--- 验证数据集列名 ---")
print("数据集特征:", dataset.features)
if dataset:
    print("第一个样本的键:", dataset[0].keys())
print("--- 验证结束 ---")

# 应用格式化函数
print("\n正在应用格式化函数...")
dataset = dataset.map(formatting_prompts_func, batched = True,)
print("格式化函数应用成功。")
print("\n数据准备完成。格式化文本示例:")
print(dataset[0]["text"])


# --- 4. 模型训练 ---
print("\n--- 开始模型训练 ---")
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2, # 使用2个进程并行处理数据
    packing = False, # 是否打包短序列，对于较长序列设为False
    args = TrainingArguments(
        per_device_train_batch_size = 2, # G5.4xlarge (A10G 24GB) 可以尝试增大此值 (如 4 或 8)
        gradient_accumulation_steps = 4, # 梯度累积步数，有效批次大小 = batch_size * accumulation_steps
        warmup_steps = 5,                # 预热步数
        max_steps = 60,                  # 快速测试，设为 60 步。完整训练请注释掉此行，并设置 num_train_epochs=1
        # num_train_epochs = 1,          # 或者设置训练的总轮数
        learning_rate = 2e-4,            # 学习率
        fp16 = not is_bfloat16_supported(), # 如果不支持 bfloat16，则使用 fp16
        bf16 = is_bfloat16_supported(),     # 如果支持 bfloat16，则使用它 (A10G 支持)
        logging_steps = 1,               # 每隔多少步记录一次日志 (设为1可清晰看到进度)
        optim = "adamw_8bit",            # 使用8位 AdamW 优化器节省显存
        weight_decay = 0.01,             # 权重衰减
        lr_scheduler_type = "linear",    # 学习率调度器类型
        seed = 3407,                     # 随机种子
        output_dir = "outputs",          # 训练输出（检查点等）保存目录
        report_to = "none",              # 不报告给 WandB 或 TensorBoard (可选 "wandb", "tensorboard")
    ),
)

# 显示当前显存状态
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU 型号 = {gpu_stats.name}. 总显存 = {max_memory} GB.")
print(f"训练开始前已预留显存 = {start_gpu_memory} GB.")

# 开始训练
trainer_stats = trainer.train()

# 显示最终显存和时间统计
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
print(f"\n--- 训练完成 ---")
print(f"训练耗时: {trainer_stats.metrics['train_runtime']:.2f} 秒")
print(f"训练耗时: {trainer_stats.metrics['train_runtime']/60:.2f} 分钟")
print(f"峰值显存占用 = {used_memory} GB.")
print(f"训练使用的峰值显存 = {used_memory_for_lora} GB.")
print(f"峰值显存占用率 = {used_percentage} %.")
print(f"训练使用的峰值显存占用率 = {lora_percentage} %.")


# --- 5. 推理测试 ---
print("\n--- 开始推理测试 ---")
FastLanguageModel.for_inference(model) # 启用 Unsloth 的快速推理模式

# --- 测试用例 1 ---
instruction_test_1 = "命盤中破軍星在子女宮，對生育有影響嗎"
input_test_1 = "" # 输入字段在我们的格式中是空的

inputs_1 = tokenizer(
    [
        alpaca_prompt.format(
            instruction_test_1,
            input_test_1,
            "" # Response 留空以生成
        )
    ], return_tensors = "pt").to("cuda")

print(f"\n--- 测试用例 1 (标准生成): '{instruction_test_1}' ---")
outputs_1 = model.generate(**inputs_1, max_new_tokens = 256, use_cache = True) # 增加生成长度
decoded_output_1 = tokenizer.batch_decode(outputs_1, skip_special_tokens=True)[0]
print("模型输出:")
print(decoded_output_1)

# 提取生成的响应部分
response_marker = "### Response:"
response_start_index = decoded_output_1.find(response_marker)
if response_start_index != -1:
    generated_text_1 = decoded_output_1[response_start_index + len(response_marker):].strip()
    print("\n--- 仅生成的响应 ---")
    print(generated_text_1)
    print("--- ----- ---")


# --- 测试用例 2 (使用您提供的示例问题) ---
instruction_test_2 = "最近事业不顺，想请大师看看我的八字运势。我是1985年12月25日凌晨2点出生的?"
input_test_2 = ""

inputs_2 = tokenizer(
    [
        alpaca_prompt.format(
            instruction_test_2,
            input_test_2,
            "" # Response 留空以生成
        )
    ], return_tensors = "pt").to("cuda")

# 使用流式输出 (TextStreamer)
text_streamer = TextStreamer(tokenizer, skip_prompt=True) # skip_prompt=True 只显示生成的部分
print(f"\n--- 测试用例 2 (流式生成): '{instruction_test_2}' ---")
print("模型输出 (流式):")
_ = model.generate(**inputs_2, streamer = text_streamer, max_new_tokens = 768, use_cache = True) # 允许更长的输出
print("\n(流式生成结束)")


# --- 6. 保存模型到 S3 (可选) ---
print("\n--- 模型保存选项 ---")
save_to_s3 = input("微调完成。是否要将 LoRA 模型保存到 S3? (Y/N): ").strip().upper()

if save_to_s3 == 'Y':
    s3_path = "s3://cd4ml-ai-lab-eu-central-1-test-r2d2/llm_finetune_poc_20250417/"
    local_temp_path = "./temp_lora_model_for_s3" # 本地临时保存路径

    print(f"\n准备将模型保存到 S3 路径: {s3_path}")
    print(f"首先保存到本地临时目录: {local_temp_path}")

    # 1. 保存到本地临时目录
    os.makedirs(local_temp_path, exist_ok=True)
    model.save_pretrained(local_temp_path)
    tokenizer.save_pretrained(local_temp_path)
    print("模型和分词器已保存到本地临时目录。")

    # 2. 上传到 S3
    try:
        print("正在上传到 S3...")
        s3 = boto3.client('s3')
        parsed_s3_path = urlparse(s3_path)
        bucket_name = parsed_s3_path.netloc
        # 确保存储桶中的对象键以 '/' 结尾，如果前缀不为空的话
        s3_prefix = parsed_s3_path.path.lstrip('/')
        if s3_prefix and not s3_prefix.endswith('/'):
            s3_prefix += '/'

        for root, dirs, files in os.walk(local_temp_path):
            for filename in files:
                local_file_path = os.path.join(root, filename)
                # 计算相对路径作为 S3 中的对象键
                relative_path = os.path.relpath(local_file_path, local_temp_path)
                s3_key = os.path.join(s3_prefix, relative_path).replace("\\", "/") # 兼容 Windows 路径

                print(f"  上传文件: {local_file_path} 到 s3://{bucket_name}/{s3_key}")
                s3.upload_file(local_file_path, bucket_name, s3_key)

        print(f"模型成功上传到 S3: {s3_path}")

        # 3. 清理本地临时文件
        print(f"正在删除本地临时目录: {local_temp_path}")
        shutil.rmtree(local_temp_path)
        print("本地临时目录已删除。")

    except Exception as e:
        print(f"上传到 S3 失败: {e}")
        print("请检查 AWS 凭证配置、S3 存储桶权限以及网络连接。")
        print(f"本地临时文件保留在: {local_temp_path}")

else:
    print("选择不保存模型到 S3。")
    # 如果仍然需要本地副本，可以取消注释下面两行
    # print("将模型保存到本地目录 'lora_model'...")
    # model.save_pretrained("lora_model")
    # tokenizer.save_pretrained("lora_model")
    # print("模型已保存到本地 'lora_model' 目录。")


# --- 可选：保存其他格式的代码（保持注释状态）---
# 以下是保存为其他格式（如 merged 16bit, 4bit 或 GGUF）的示例代码
# 如果需要，可以取消注释并根据需要进行修改

# # Merge to 16bit
# if False:
#    print("Merging model to 16bit and saving locally...")
#    model.save_pretrained_merged("merged_16bit_model", tokenizer, save_method = "merged_16bit",)
#    print("Model saved in 16bit format to 'merged_16bit_model'.")
# # Merge to 4bit
# if False:
#    print("Merging model to 4bit and saving locally...")
#    model.save_pretrained_merged("merged_4bit_model", tokenizer, save_method = "merged_4bit",)
#    print("Model saved in 4bit format to 'merged_4bit_model'.")
# # Save GGUF
# if False:
#    print("Saving model in GGUF Q8_0 format...")
#    model.save_pretrained_gguf("model_q8_0", tokenizer, quantization_method = "q8_0")
#    print("Model saved in GGUF Q8_0 format to 'model_q8_0'.")
# if False:
#    print("Saving model in GGUF Q4_K_M format...")
#    model.save_pretrained_gguf("model_q4_k_m", tokenizer, quantization_method = "q4_k_m")
#    print("Model saved in GGUF Q4_K_M format to 'model_q4_k_m'.")

print("\n脚本执行完毕。")