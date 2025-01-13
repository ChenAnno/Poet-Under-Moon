import os
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from dataset import PoemDataset
from model import TransformerPoem
import argparse
from tqdm import tqdm  # 导入 tqdm
import random
import numpy as np


def set_seed(seed):
    """固定随机数种子"""
    random.seed(seed)          # Python 随机数种子
    np.random.seed(seed)       # NumPy 随机数种子
    torch.manual_seed(seed)    # PyTorch 随机数种子
    torch.cuda.manual_seed(seed)  # CUDA 随机数种子
    torch.cuda.manual_seed_all(seed)  # 如果使用多卡，设置所有 GPU 的随机数种子
    torch.backends.cudnn.deterministic = True  # 确保 CUDA 卷积操作的结果确定
    torch.backends.cudnn.benchmark = False     # 关闭 CUDA 卷积优化，以确保结果可复现


def setup(rank, world_size):
    """初始化进程组"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '50001'  # 使用一个较大的端口号
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    """清理进程组"""
    dist.destroy_process_group()

def train(rank, world_size, args):
    """训练函数"""
    setup(rank, world_size)

    set_seed(args.seed)

    # TODO 加载数据
    dataset = PoemDataset('archive/5yan.txt')
    # dataset = PoemDataset('archive/chinese_poems.txt')
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler)

    # 初始化模型
    model = TransformerPoem(vocab_size=len(dataset.word2idx)).to(rank)
    model = DDP(model, device_ids=[rank])

    # 训练逻辑
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.word2idx['<pad>'])

    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        model.train()

        # 使用 tqdm 显示训练进度条
        dataloader_len = len(dataloader)
        if rank == 0 and args.tqdm:  # 只在主进程显示进度条
            dataloader = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.epochs}", leave=False, total=dataloader_len)

        for batch in dataloader:
            src = batch.to(rank)  # 完整的输入序列
            tgt = batch.to(rank)  # 完整的目标序列
            # print("Here:", src.shape, tgt.shape)
            # 这里太容易写错了，务必仔细检查
            output = model(src[:, :-1], tgt[:, :-1])

            # 计算损失（解码器的输出需要与 tgt 的后 n-1 个 token 进行比较）
            loss = criterion(output.view(-1, output.size(-1)), tgt[:, 1:].reshape(-1))

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if rank == 0 and args.tqdm:  # 只在主进程打印日志
                dataloader.set_postfix(loss=loss.item())

        # 每个epoch结束后生成诗歌（只在主进程生成）
        if rank == 0:
            start_text = "窗前明月光"
            generated_poem = generate(model.module, start_text, dataset)
            print(f"\nEpoch {epoch + 1}, 生成诗歌: {generated_poem}")

    cleanup()

def generate(model, start_text, dataset, max_len=50, temperature=1.0):
    """自回归生成诗歌"""
    model.eval()
    tokens = list(start_text)
    tokens = ['<bos>'] + tokens
    input_ids = [dataset.word2idx.get(token, dataset.word2idx['<unk>']) for token in tokens]
    input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).cuda()
    generated_ids = input_ids
    for _ in range(max_len):
        with torch.no_grad():
            output = model(generated_ids, generated_ids)
            logits = output[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)

            if next_token.item() == dataset.word2idx['<eos>']:
                break

    generated_tokens = [dataset.idx2word[idx.item()] for idx in generated_ids[0]]
    return ''.join(generated_tokens[1:-1])  # 去掉<bos>和<eos>

if __name__ == '__main__':
    # 解析参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--world_size', type=int, default=8, help='Number of GPUs')
    parser.add_argument('--epochs', type=int, default=5000, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--tqdm', type=bool, default=False, help='Whether use tqdm')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    # 启动多进程训练
    torch.multiprocessing.spawn(
        train,
        args=(args.world_size, args),
        nprocs=args.world_size,
        join=True
    )
