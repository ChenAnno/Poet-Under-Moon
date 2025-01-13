import torch
from torch.utils.data import Dataset
from collections import defaultdict

class PoemDataset(Dataset):
    def __init__(self, file_path, max_len=128):
        self.data = []
        self.word2idx = defaultdict(lambda: len(self.word2idx))
        self.idx2word = {}
        self.max_len = max_len
        # 特殊token
        self.word2idx['<pad>'] = 0
        self.word2idx['<bos>'] = 1
        self.word2idx['<eos>'] = 2
        self.word2idx['<unk>'] = 3
        # 加载数据
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    # 分词（简单按字分割）
                    tokens = list(line)
                    tokens = ['<bos>'] + tokens + ['<eos>']
                    tokens = tokens[:self.max_len]  # 截断
                    tokens += ['<pad>'] * (self.max_len - len(tokens))  # 填充
                    self.data.append(tokens)
                # print("Line:", line)
        # 构建词汇表
        for tokens in self.data:
            for token in tokens:
                if token not in self.word2idx:
                    self.word2idx[token] = len(self.word2idx)
        self.idx2word = {v: k for k, v in self.word2idx.items()}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens = self.data[idx]
        ids = [self.word2idx[token] for token in tokens]
        return torch.tensor(ids, dtype=torch.long)

# 测试数据加载
if __name__ == '__main__':
    dataset = PoemDataset('archive/chinese_poems.txt') # 一共304752首诗， https://www.kaggle.com/datasets/qianboao/chinesepoetrydataset
    print(f"词汇表大小: {len(dataset.word2idx)}")
    print(f"样例数据: {dataset[0]}")
