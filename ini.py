from datasets import load_dataset, Dataset

load_dataset('openai/gsm8k', 'main')
# 之所以单独做了这个，是因为服务器莫名其妙的镜像问题，懒得解决了，这样放在sh里执行一步到位
