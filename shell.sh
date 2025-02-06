sh requirment.sh
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download --resume-download Qwen/Qwen2.5-1.5B-Instruct --local-dir Qwen/Qwen2.5-1.5B-Instruct
git clone https://hf-mirror.com/datasets/openai/gsm8k
# huggingface-cli download --resume-download deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --local-dir deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
python ini.py
python sft.py
python rl-zero.py
python rl-sft.py
python test.py
