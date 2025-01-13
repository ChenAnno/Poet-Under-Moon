import argparse
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.optim as optim
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, random_split
from rich.progress import Progress
from dataset import PoetryData
from model import PoetryNet
import time

batch_size = 64
lr = 0.0001

class PoetryGen:
    def __init__(self, rank, world_size, args) -> None:
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(f"cuda:{rank}")
        self.dataset = PoetryData(self.device, max_lines=50000, token_length=12)
        self.vocab_size = self.dataset.vocab_size
        train_data, test_data = random_split(self.dataset, [len(self.dataset) - 1000, 1000])
        train_sampler = DistributedSampler(train_data, num_replicas=world_size, rank=rank)
        test_sampler = DistributedSampler(test_data, num_replicas=world_size, rank=rank)
        self.train_dataloader = DataLoader(train_data, batch_size, sampler=train_sampler)
        self.test_dataloader = DataLoader(test_data, batch_size, sampler=test_sampler)

        self.net = PoetryNet(self.vocab_size, self.device, embed_size=512).to(self.device)
        self.net = DDP(self.net, device_ids=[rank])
        self.optimizer = optim.Adam(self.net.parameters(), lr)
        self.optimizer_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, 256)

        self.loss_f = torch.nn.CrossEntropyLoss(ignore_index=2)
        self.loaded_checkpoint_file = None
        self.epoch = 0

        self.progress = Progress()

        import glob

        files = glob.glob("checkpoint-*.pth")
        for i, file in enumerate(files):
            print(f"{i}> {file}")
        if files:
            t = input("choose check point to load, default is the last one, n to unload>")
            if t == "":
                t = -1
            if t != "n":
                self.load_checkpoint(files[int(t)])

    def save_checkpoint(self):
        file_name = (
            self.loaded_checkpoint_file
            or f'checkpoint-{time.strftime("%y%m%d-%H%M")}.pth'
        )
        if self.rank == 0:
            with open(file_name, "wb") as file:
                torch.save(
                    {
                        "net_state": self.net.module.state_dict(),
                        "optimizer_state": self.optimizer.state_dict(),
                        "epoch": self.epoch,
                    },
                    file,
                )
            print(f"save check point to {file_name}")
            self.loaded_checkpoint_file = file_name

    def load_checkpoint(self, file: str):
        ckpt = torch.load(file, map_location=self.device)
        self.net.load_state_dict(ckpt["net_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.epoch = ckpt["epoch"]

        self.loaded_checkpoint_file = file
        self.optimizer_scheduler.last_epoch = self.epoch
        print(f"loaded check point: {file}, epoch: {self.epoch}")

    def generate_one(self, pre_sentence: str, start_words: str = ""):
        self.net.eval()
        start_words_token = [0]
        start_words_token.extend(self.dataset.word2idx[x] for x in start_words)
        src = self.dataset.word2token(pre_sentence).unsqueeze(0)
        tgt = torch.LongTensor([start_words_token]).to(self.device)
        memo = self.net.module.encode(src)
        res = []
        for i in range(12):
            out = self.net.module.decode(tgt, memo)
            next_word = out.argmax(2)
            if next_word[0][-1] == 1:
                break
            res.append(next_word[0][-1].item())
            tgt = torch.cat((tgt, next_word[:, -1:]), 1)

        return start_words + self.dataset.token2word(res)

    def generate(self, num_sentence: int, pre_style: str):
        res = []
        for i in range(num_sentence):
            s = self.generate_one(pre_style if not res else res[-1])
            res.append(s)
        return "/".join(res)

    def generate_by_start(self, start_words: str, pre_style: str) -> str:
        res = []
        start_words_l = start_words.split("/")
        if not start_words_l:
            return ""
        for i, s in enumerate(start_words_l):
            t = self.generate_one(pre_style if not res else res[-1], s)
            res.append(t)
        return "/".join(res)

    def forward_net(self, src: Tensor, tgt: Tensor):
        src, tgt = src.to(self.device), tgt.to(self.device)
        src_mask = (src == 2).to(self.device)

        dec_tgt = tgt[:, :-1]
        dec_tgt_mask = (dec_tgt == 2).to(self.device)
        tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(dec_tgt.size(1), self.device)

        out = self.net(src, dec_tgt, tgt_mask, src_mask, dec_tgt_mask)
        return out

    def train_epoch(self):
        self.net.train()
        train_progress = self.progress.add_task(description="Train Epoch", total=len(self.train_dataloader))
        loss_f = self.loss_f

        vocab_size = self.dataset.vocab_size
        len_data = len(self.train_dataloader)
        loss_all = 0
        for i, (src, tgt) in enumerate(self.train_dataloader):
            out = self.forward_net(src, tgt)
            loss = loss_f(out.reshape(-1, vocab_size), tgt[:, 1:].flatten())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.progress.update(
                train_progress,
                advance=1,
                description=f"{i}/{len_data} loss={loss.item():.4f}",
            )
            loss_all += loss.item()
        self.optimizer_scheduler.step()
        self.progress.remove_task(train_progress)
        self.progress.print(
            f"train epoch={self.epoch} average loss={loss_all/len_data:.4f} lr={self.optimizer_scheduler.get_lr()}"
        )

    def evaluation(self):
        self.net.eval()

        loss_f = self.loss_f
        vocab_size = self.dataset.vocab_size

        loss_a = 0
        with torch.no_grad():
            for i, (src, tgt) in enumerate(self.test_dataloader):
                out = self.forward_net(src, tgt)
                loss = loss_f(out.reshape(-1, vocab_size), tgt[:, 1:].flatten())
                loss_a += loss.item()

        self.progress.print(
            f"Validation: epoch={self.epoch} avg loss={loss_a/len(self.test_dataloader):.4f}"
        )

    def training(self, train_epoch_nums: int = 36):
        self.progress.start()
        training_all = self.progress.add_task(
            description=f"epoch={self.epoch} lr={self.optimizer_scheduler.get_lr()}",
            total=train_epoch_nums,
        )
        for i in range(train_epoch_nums):
            self.progress.update(
                training_all,
                advance=1,
                description=f"epoch={self.epoch} lr={self.optimizer_scheduler.get_lr()}",
            )
            self.train_epoch()
            self.evaluation()
            self.epoch += 1
            self.save_checkpoint()
            print(self.generate(4, "床前明月光"))

def parse_args():
    parser = argparse.ArgumentParser(description="Transformer Model Training with DDP")
    
    parser.add_argument('--file_path', type=str, default='archive/chinese_poems.txt', help='Path to the training data')
    parser.add_argument('--seq_length', type=int, default=50, help='Sequence length for training')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size for training')
    parser.add_argument('--d_model', type=int, default=512, help='Dimension of model')
    parser.add_argument('--nhead', type=int, default=8, help='Number of heads in the multiheadattention models')
    parser.add_argument('--num_encoder_layers', type=int, default=6, help='Number of encoder layers')
    parser.add_argument('--num_decoder_layers', type=int, default=6, help='Number of decoder layers')
    parser.add_argument('--dim_feedforward', type=int, default=2048, help='Dimension of the feedforward network model')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout value')
    parser.add_argument('--activation', type=str, default='relu', choices=['relu', 'gelu', 'glu'], help='Activation function to use')
    parser.add_argument('--use_relative_positions', action='store_true', help='Use relative position encoding')
    parser.add_argument('--use_sparse_attention', action='store_true', help='Use sparse attention mechanism')
    parser.add_argument('--epochs', type=int, default=36, help='Number of epochs to train')
    parser.add_argument('--save_model', type=str, default='transformer_model.pth', help='Path to save the trained model')
    parser.add_argument('--world_size', type=int, default=8, help='Number of GPUs to use')
    parser.add_argument('--init_lr', type=float, default=0.008, help='Initial learning rate')
    parser.add_argument('--val_split', type=float, default=0.2, help='Validation data split ratio')

    return parser.parse_args()

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.manual_seed(42)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def main_worker(rank, world_size, args):
    setup(rank, world_size)

    model = PoetryGen(rank, world_size, args)
    model.training(args.epochs)

    cleanup()

def main():
    args = parse_args()
    world_size = args.world_size
    mp.spawn(main_worker, args=(world_size, args), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()