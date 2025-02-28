from pathlib import Path

import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT_DIR = Path(__file__).parent
data_path = os.path.join(ROOT_DIR, '../test.log')

# Bert_path = r"/mnt/workspace/.cache/modelscope/hub/models/AI-ModelScope/bert-base-uncased"
Qwen_path = r"/mnt/workspace/.cache/modelscope/models/Qwen/Qwen2___5-1___5B-Instruct"
# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(
    f'dataset_name: {data_path}\n'
    f'Qwen_path: {Qwen_path}\n'
    f'device: {device}')

# 加载分词器和模型
# 这里使用 Qwen-7B-Chat 模型，你可以根据需要更换为其他 Qwen 系列模型
tokenizer = AutoTokenizer.from_pretrained(Qwen_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(Qwen_path, trust_remote_code=True, low_cpu_mem_usage=True,
                                             device_map=device).eval()

# 输入文本
input_text = "介绍一下北京的故宫。"

# 对输入文本进行分词
inputs = tokenizer(input_text, return_tensors='pt').to(model.device)

# 进行推理
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=512, do_sample=True, top_p=0.85, temperature=0.35)

# 将输出的 token 序列转换为文本
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

# 打印生成的回复
print("输入问题:", input_text)
print("Qwen 模型的回复:", response)
