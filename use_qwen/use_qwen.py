import time
from pathlib import Path

import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT_DIR = Path(__file__).parent
data_path = os.path.join(ROOT_DIR, '../test.log')

# Bert_path = r"/mnt/workspace/.cache/modelscope/hub/models/AI-ModelScope/bert-base-uncased"
Qwen_path = r"/mnt/workspace/.cache/modelscope/hub/models/Qwen/Qwen2.5-1.5B-Instruct"
# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    # 以只读模式打开 test.log 文件
    with open(data_path, 'r', encoding='utf-8') as file:
        # 读取文件的全部内容
        content = file.read()
        # 按换行符分割内容成列表
        lines = content.split('\n')
except FileNotFoundError:
    print("未找到 test.log 文件，请检查文件路径是否正确。")
except Exception as e:
    print(f"读取文件时出现错误: {e}")

print(
    f'dataset_name: {data_path}\n'
    f'Qwen_path: {Qwen_path}\n'
    f'lines: {len(lines)}\n'
    f'device: {device}')

# 加载分词器和模型
# 这里使用 Qwen-7B-Chat 模型，你可以根据需要更换为其他 Qwen 系列模型
tokenizer = AutoTokenizer.from_pretrained(Qwen_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(Qwen_path, trust_remote_code=True, low_cpu_mem_usage=True,
                                             device_map=device).eval()


def eval(input):
    # 对输入文本进行分词
    inputs = tokenizer(input, return_tensors='pt').to(model.device)

    # 进行推理
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=512, do_sample=True, top_p=0.85, temperature=0.35)

    # 将输出的 token 序列转换为文本
    llm_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return llm_response


# 输入文本
input_text = "介绍一下北京的故宫。"
response = eval(input_text)
# 打印生成的回复
print("输入问题:", )
print("Qwen 模型的回复:", response)

start_time = time.time()

for line in lines:
    input_text = "给你提供一段日志，请帮我对日志进行分类，分类的结果只有两种，正常，异常，请直接高速我分类的结果，不要返回其它信息。日志：" + line
    response = eval(input_text)
    print(f"模型输出：", response)

end_time = time.time()

# 计算运行耗时
elapsed_time = end_time - start_time
print(f"推理耗时: {elapsed_time} 秒")
