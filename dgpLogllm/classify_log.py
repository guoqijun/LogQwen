import os
from pathlib import Path
import torch
from dataset import Dataset
from dgpLogModel import DgpLogModel

max_content_len = 512

ROOT_DIR = Path(__file__).parent
data_path = os.path.join(ROOT_DIR, '../test.log')

Bert_path = r"/mnt/workspace/.cache/modelscope/hub/models/AI-ModelScope/bert-base-uncased"
Qwen_path = r"/mnt/workspace/.cache/modelscope/hub/models/Qwen/Qwen2___5-1___5B-Instruct"

device = torch.device("cuda:0")

print(
    f'dataset_name: {data_path}\n'
    f'max_content_len: {max_content_len}\n'
    f'Bert_path: {Bert_path}\n'
    f'Qwen_path: {Qwen_path}\n'
    f'device: {device}')


def classify(model, dataset):
    model.eval()
    # 打印分割后的列表
    # 遍历列表并打印每一行
    print("给model的数据", dataset.sequences)
    outputs_ids = model(dataset.sequences)
    print("推理后：", outputs_ids)
    outputs = model.Llama_tokenizer.batch_decode(outputs_ids)
    print("编码后：", outputs)

    print("fuck you")


if __name__ == '__main__':
    print(f'dataset: {data_path}')
    dataset = Dataset(data_path)
    model = DgpLogModel(Bert_path, Qwen_path, device=device, max_content_len=max_content_len)
    classify(model, dataset)
