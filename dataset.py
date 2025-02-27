import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import re

patterns = [
    r'True',
    r'true',
    r'False',
    r'false',
    r'\b(zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|billion)\b',
    r'\b(Mon|Monday|Tue|Tuesday|Wed|Wednesday|Thu|Thursday|Fri|Friday|Sat|Saturday|Sun|Sunday)\b',
    r'\b(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+(\d{1,2})\s+\b',
    r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(:\d{1,5})?',  # IP
    r'([0-9A-Fa-f]{2}:){11}[0-9A-Fa-f]{2}',  # Special MAC
    r'([0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}',  # MAC
    r'[a-zA-Z0-9]*[:\.]*([/\\]+[^/\\\s\[\]]+)+[/\\]*',  # File Path
    r'\b[0-9a-fA-F]{8}\b',
    r'\b[0-9a-fA-F]{10}\b',
    r'(\w+[\w\.]*)@(\w+[\w\.]*)\-(\w+[\w\.]*)',
    r'(\w+[\w\.]*)@(\w+[\w\.]*)',
    r'[a-zA-Z\.\:\-\_]*\d[a-zA-Z0-9\.\:\-\_]*',  # word have number
]

# 合并所有模式
combined_pattern = '|'.join(patterns)


# 替换函数
def replace_patterns(text):
    text = re.sub(r'[\.]{3,}', '.. ', text)  # Replace multiple '.' with '.. '
    text = re.sub(combined_pattern, '<*>', text)
    return text


class Dataset(Dataset):
    def __init__(self, file_path):
        try:
            # 以只读模式打开 test.log 文件
            with open('test.log', 'r', encoding='utf-8') as file:
                # 读取文件的全部内容
                content = file.read()
                # 按换行符分割内容成列表
                lines = content.split('\n')
                # 打印分割后的列表
                # 遍历列表并打印每一行
                for line in lines:
                    print(line)
        except FileNotFoundError:
            print("未找到 test.log 文件，请检查文件路径是否正确。")
        except Exception as e:
            print(f"读取文件时出现错误: {e}")

        self.sequences = lines

    def __len__(self):
        return len(self.sequences)

    def get_batch(self, indexes):
        this_batch_seqs = self.sequences[indexes]
        return this_batch_seqs
