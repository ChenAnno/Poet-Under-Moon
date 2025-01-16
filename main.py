import os
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from dataset import PoemDataset
from model import TransformerPoem, TransformerPoemCustom
import argparse
from tqdm import tqdm
import random
import numpy as np
import matplotlib.pyplot as plt
import json
import time


def set_seed(seed):
    """固定随机数种子"""
    random.seed(seed)  # Python 随机数种子
    np.random.seed(seed)  # NumPy 随机数种子
    torch.manual_seed(seed)  # PyTorch 随机数种子
    torch.cuda.manual_seed(seed)  # CUDA 随机数种子
    torch.cuda.manual_seed_all(seed)  # 如果使用多卡，设置所有 GPU 的随机数种子
    torch.backends.cudnn.deterministic = True  # 确保 CUDA 卷积操作的结果确定
    torch.backends.cudnn.benchmark = False  # 关闭 CUDA 卷积优化，以确保结果可复现


def setup(rank, world_size):
    """初始化进程组"""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "50011"  # 使用一个较大的端口号
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    """清理进程组"""
    dist.destroy_process_group()


def train(rank, world_size, args):
    """训练函数"""
    setup(rank, world_size)

    # 加载数据
    train_dataset = PoemDataset(args.train_data_path)
    sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler)

    # 初始化模型
    if not args.use_custom:
        model = TransformerPoem(
            vocab_size=len(train_dataset.word2idx),
            use_glu=args.use_glu,
            use_relative_pos=args.use_relative_pos,
            use_sparse_attn=args.use_sparse_attn,
            residual_before_ln=args.residual_before_ln,
        ).to(rank)
    else:
        model = TransformerPoemCustom(
            vocab_size=len(train_dataset.word2idx),
            use_glu=args.use_glu,
            use_relative_pos=args.use_relative_pos,
            use_sparse_attn=args.use_sparse_attn,
            residual_before_ln=args.residual_before_ln,
        ).to(rank)
    model = DDP(model, device_ids=[rank])

    # 训练逻辑
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=train_dataset.word2idx["<pad>"])

    # 记录损失
    losses = []
    start_time = time.time()
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        model.train()

        # 使用 tqdm 显示训练进度条
        if rank == 0 and args.tqdm:  # 只在主进程显示进度条
            dataloader = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}", leave=False, total=len(dataloader))

        epoch_loss = 0
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

            epoch_loss += loss.item()
            if rank == 0 and args.tqdm:  # 只在主进程打印日志
                dataloader.set_postfix(loss=loss.item())

        # 计算平均损失
        epoch_loss /= len(dataloader)
        losses.append(epoch_loss)

        # 每个epoch结束后生成诗歌（只在主进程生成）
        if rank == 0:
            start_text = "窗前明月光"
            generated_poem = generate(model.module, start_text, train_dataset)
            print(f"\nEpoch {epoch}, 生成诗歌: {generated_poem}")

    if rank == 0:
        # 运行时间
        end_time = time.time()
        execution_time_minutes = (end_time - start_time) / 60
        print(f"训练{args.exp_name}耗时: {execution_time_minutes:.2f} 分钟")

        # 保存模型 ckpt
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.module.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": epoch_loss,
        }
        torch.save(checkpoint, f"ckpt/{args.exp_name}_final.pt")
        print(f"模型已保存为 ckpt/{args.exp_name}_final.pt")

        # 保存损失曲线
        with open("losses.json", "w") as f:
            json.dump(losses, f)  # 保存损失数据
        plt.plot(losses, label="Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss Curve")
        plt.legend()
        plt.savefig(f"ckpt/{args.exp_name}_loss_curve.pdf")  # 保存损失曲线图
        print("损失曲线已保存为 loss_curve.pdf")

        # 简单测试
        avg_loss, perplexity = evaluate(model, train_dataset, model.device)
        print(f"评估结果 - 平均损失: {avg_loss:.4f}, 困惑度: {perplexity:.4f}")

    cleanup()


def generate(model, start_text, dataset, max_len=50, temperature=1.0):
    """自回归生成诗歌"""
    model.eval()
    tokens = list(start_text)
    tokens = ["<bos>"] + tokens
    input_ids = [dataset.word2idx.get(token, dataset.word2idx["<unk>"]) for token in tokens]
    input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).cuda()
    generated_ids = input_ids
    for _ in range(max_len):
        with torch.no_grad():
            output = model(generated_ids, generated_ids)
            logits = output[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)

            if next_token.item() == dataset.word2idx["<eos>"]:
                break

    generated_tokens = [dataset.idx2word[idx.item()] for idx in generated_ids[0]]
    return "".join(generated_tokens[1:-1])  # 去掉<bos>和<eos>


def evaluate(model, dataset, device):
    """评估模型性能"""
    """
    base: 评估结果 - 平均损失: 0.9386, 困惑度: 2.5563
    glu: 评估结果 - 平均损失: 0.9383, 困惑度: 2.5555
    relative_pos: 评估结果 - 平均损失: 0.9412, 困惑度: 2.5631
    sparse: 评估结果 - 平均损失: 0.1119, 困惑度: 1.1185
    before: 评估结果 - 平均损失: 1.2684, 困惑度: 3.5552
    custom: 评估结果 - 平均损失: 0.9388, 困惑度: 2.5569
    """
    model.eval()
    total_loss = 0
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.word2idx["<pad>"])

    # 加载验证集或测试集
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    with torch.no_grad():
        for batch in dataloader:
            src = batch.to(device)
            tgt = batch.to(device)
            output = model(src[:, :-1], tgt[:, :-1])
            loss = criterion(output.view(-1, output.size(-1)), tgt[:, 1:].reshape(-1))
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    perplexity = torch.exp(torch.tensor(avg_loss)).item()  # 计算困惑度
    return avg_loss, perplexity


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--world_size", type=int, default=8, help="Number of GPUs")
    parser.add_argument("--train_data_path", type=str, default="./archive/5yan.txt", help="Path to the train data")
    parser.add_argument("--tqdm", type=bool, default=False, help="Whether use tqdm")

    parser.add_argument("--use_glu", type=bool, default=False, help="vanilla: False")
    parser.add_argument("--use_relative_pos", type=bool, default=False, help="vanilla: False")
    parser.add_argument("--use_sparse_attn", type=bool, default=False, help="vanilla: False")
    parser.add_argument("--residual_before_ln", type=bool, default=True, help="vanilla: True")
    parser.add_argument("--num_layers", type=int, default=12, help="vanilla: 12")
    parser.add_argument("--use_custom", type=bool, default=True, help="self-constructed")

    parser.add_argument("--exp_name", type=str, default="custom", help="experiment name")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")

    args = parser.parse_args()

    # 固定随机数种子
    set_seed(args.seed)

    # 启动多进程训练
    torch.multiprocessing.spawn(train, args=(args.world_size, args), nprocs=args.world_size, join=True)
