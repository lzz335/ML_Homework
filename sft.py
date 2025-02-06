import json
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
import torch
from sklearn.model_selection import train_test_split

DATASET_PATH = "sft_dataset.json"


def process_data(entries):
    processed = []
    for i in range(len(entries)):
        query = entries[i]
        user_entry = query[0]
        assistant_entry = query[1]

        # 确保包含所需的键
        if "user" in user_entry and "content" in assistant_entry:
            messages = [
                {"role": "user", "content": user_entry["user"]},
                {"role": "system", "content": assistant_entry["content"]}
            ]
            processed.append({"messages": messages})
    return processed


def load_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    processed = process_data(data)
    train_data, val_data = train_test_split(processed, test_size=0.05, random_state=42)

    return Dataset.from_list(train_data), Dataset.from_list(val_data)


model_name = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
    trust_remote_code=True
)


def format_dataset(example):
    return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False)}


training_args = TrainingArguments(
    output_dir="./qwen_finetuned",
    num_train_epochs=5,  # 训练的轮数
    per_device_train_batch_size=4,  # 训练批次大小（根据GPU显存调整）
    per_device_eval_batch_size=1,  # 验证批次大小
    gradient_accumulation_steps=4,  # 梯度累积步数，可以增加有效的批次大小
    optim="adamw_torch",  # 优化器
    learning_rate=1e-5,  # 学习率
    weight_decay=0.01,  # 权重衰减
    warmup_ratio=0.03,  # 预热比例
    lr_scheduler_type="cosine",  # 学习率调度器类型
    logging_steps=10,  # 日志记录步数
    evaluation_strategy="epoch",  # 评估策略
    save_strategy="epoch",  # 保存策略
    save_total_limit=2,  # 最多保存的检查点数量
    bf16=True if torch.cuda.is_available() else False,  # 如果有GPU，则使用半精度
    report_to=["tensorboard"],  # 报告工具，可以是["tensorboard", "wandb"]
    load_best_model_at_end=True,  # 在训练结束时加载最佳模型
    metric_for_best_model="loss",  # 用于评估最佳模型的指标
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

train_dataset, eval_dataset = load_dataset(DATASET_PATH)

train_dataset = train_dataset.map(format_dataset, remove_columns=train_dataset.column_names)
eval_dataset = eval_dataset.map(format_dataset, remove_columns=eval_dataset.column_names)


def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=1024,
        padding="max_length"
    )


train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
eval_dataset = eval_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
eval_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

trainer.train()

trainer.save_model("./qwen_finetuned_final")

eval_results = trainer.evaluate()
print(f"Evaluation results: {eval_results}")
