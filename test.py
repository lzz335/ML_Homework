from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, Dataset
import re
import numpy as np
from tqdm import tqdm
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
SYSTEM_PROMPT = """
A conversation between User and Assistant. The user asks a question, and the Assistant solves it. 
The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The 
reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., 
<think> reasoning process here </think><answer> answer here </answer>. Your answer only needs the final number, no additional description is required.
"""
model_name = "Qwen/Qwen2.5-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

def get_gsm8k_questions(split = "test") -> Dataset:
    data = load_dataset('openai/gsm8k', 'main')[split]
    data = data.map(lambda x: { # type: ignore
        'prompt': tokenizer.apply_chat_template(
                        [
                    {'role': 'system', 'content': SYSTEM_PROMPT},
                    {'role': 'user', 'content': x['question']}
                ],
                tokenize=False,
                add_generation_prompt=True
            ),
        'answer': extract_hash_answer(x['answer'])
    })
    return data

dataset = get_gsm8k_questions()
data_loader = DataLoader(dataset, batch_size=64, shuffle=False)
# device = "cpu"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import os

def get_all_folders(path):
    # 使用 os.listdir 获取路径下的所有文件和文件夹
    all_items = os.listdir(path)
    # 筛选出其中的文件夹
    folders = [item for item in all_items if os.path.isdir(os.path.join(path, item))]
    new_fold = []
    for folder in folders:
        if folder != ".ipynb_checkpoints":
            new_fold.append(path + "/" + folder)
    return new_fold
path = "Qwen/SFT-Qwen2.5-1.5B-Instruct-GRPO" 
folders = get_all_folders(path)
def calculate_format_satisfy(response_data):
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    judges = []
    for res in response_data:
        judges.append(re.match(pattern, res) is not None)
    return judges

def correctness_satisfy(response_data,answers):
    judges = []
    for res,ans in zip(response_data,answers):
        str_data = extract_xml_answer(res)
        judges.append(str_data == ans)
    return judges
folder_names = folders
results = []
for folder_name in folder_names:
    model = AutoModelForCausalLM.from_pretrained(
        folder_name,
        torch_dtype=torch.float32
    )
    model.to(device)
    score = [[],[]]
    length = []
    for dum in tqdm(data_loader):
        model_inputs = tokenizer(dum['prompt'], return_tensors="pt", padding="max_length", max_length=256, truncation=True,padding_side='left').to(model.device)
    
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        score[0] = score[0] + correctness_satisfy(response, dum['answer'])
        score[1] = score[1] + calculate_format_satisfy(response)
        length_score = []
        res_l = tokenizer(response)["input_ids"]
        for le in res_l:
            length_score.append(len(le))
        length = length + length_score
    result = [folder_name, np.mean(length),np.mean(score[0]),np.mean(score[1]),np.std(score[0]),np.std(score[1])]
    print(result)
    results.append(result)
import pandas as pd
df = pd.DataFrame(results)
print(df)
df.to_csv("results.csv")