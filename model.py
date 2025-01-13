import math
import torch
from torch import nn as nn
from torch import Tensor, device


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, *, dropout: float = 0.1, max_len: int = 500):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # 创建位置编码矩阵
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor):
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class PoetryNet(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        device: device,
        *,
        embed_size: int = 512,
        n_head=8,
        n_layer=4,
        hidden_size=512
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size, device=device)
        self.embed_size = embed_size
        self.sq = math.sqrt(self.embed_size)
        self.transformer = nn.Transformer(
            embed_size,
            nhead=n_head,
            num_decoder_layers=n_layer,
            num_encoder_layers=n_layer,
            batch_first=True,
            dim_feedforward=hidden_size,
            device=device,
        )
        self.linear = nn.Linear(embed_size, vocab_size)
        self.positional_encoding = PositionalEncoding(embed_size)

    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        tgt_mask: Tensor = None,
        src_padding_mask: Tensor = None,
        tgt_padding_mask: Tensor = None,
    ):
        """
        Arguments:
            src: Tensor, shape ``[batch_size, seq_len]``
            tgt: Tensor, shape ``[batch_size, seq_len]``
            tgt_mask: Tensor, shape ``[tgt_seq_len, tgt_seq_len]``
            src_padding_mask: Tensor, shape ``[batch_size, src_seq_len]``
            tgt_padding_mask: Tensor, shape ``[batch_size, tgt_seq_len]``
        """
        # 对源和目标序列进行嵌入和位置编码
        src = self.embed(src) * self.sq
        src = self.positional_encoding(src)

        tgt = self.embed(tgt) * self.sq
        tgt = self.positional_encoding(tgt)

        # 通过 Transformer 模型进行编码和解码
        out = self.transformer(
            src,
            tgt,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=src_padding_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
        )
        out = self.linear(out)
        return out

    def encode(
        self, src: Tensor, noise: bool = True, noise_intensity: float = 1
    ) -> Tensor:
        """
        编码源序列
        Arguments:
            src: Tensor, shape ``[batch_size, seq_len]``
            noise: bool, 是否添加噪声
            noise_intensity: float, 噪声强度
        """
        embeded = self.embed(src)
        if noise:
            embeded += torch.rand_like(embeded) * noise_intensity
        return self.transformer.encoder(
            self.positional_encoding(embeded * self.sq)
        )

    def decode(self, tgt: Tensor, memory: Tensor) -> Tensor:
        """
        解码目标序列
        Arguments:
            tgt: Tensor, shape ``[batch_size, seq_len]``
            memory: Tensor, 编码器的输出
        """
        return self.linear(
            self.transformer.decoder(
                self.positional_encoding(self.embed(tgt) * self.sq),
                memory,
            )
        )