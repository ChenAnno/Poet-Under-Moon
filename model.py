import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer


def init_weights(module):
    """初始化模型参数"""
    if isinstance(module, nn.Linear):
        # 线性层使用 Xavier 初始化
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        # 词嵌入层使用正态分布初始化
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        # LayerNorm 的权重初始化为 1，偏置初始化为 0
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


class GLU(nn.Module):
    """GLU激活函数"""

    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim * 2)

    def forward(self, x):
        x = self.linear(x)
        return x[:, :, : x.size(-1) // 2] * torch.sigmoid(x[:, :, x.size(-1) // 2 :])


class RelativePositionEmbedding(nn.Module):
    """相对位置编码"""

    def __init__(self, max_len, d_model):
        super().__init__()
        self.embedding = nn.Embedding(2 * max_len - 1, d_model)

    def forward(self, seq_len, device):
        positions = torch.arange(seq_len, device=device).unsqueeze(0) - torch.arange(seq_len, device=device).unsqueeze(1)
        positions = positions.clamp(-seq_len + 1, seq_len - 1) + seq_len - 1
        relative_positions = self.embedding(positions)
        return relative_positions.diagonal(dim1=0, dim2=1)


class SparseAttention(nn.Module):
    """稀疏注意力机制"""

    def __init__(self, d_model, n_heads, sparsity_factor=4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.sparsity_factor = sparsity_factor
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim**-0.5

        # 定义稀疏注意力掩码
        self.register_buffer("sparse_mask", self._generate_sparse_mask(128))  # 假设最大序列长度为128

    def _generate_sparse_mask(self, seq_len):
        """生成稀疏注意力掩码"""
        mask = torch.zeros(seq_len, seq_len)
        for i in range(seq_len):
            # 每个位置只能关注到 sparsity_factor 个位置
            stride = self.sparsity_factor
            mask[i, max(0, i - stride) : min(seq_len, i + stride + 1)] = 1
        return mask.bool()

    def forward(self, q, k, v):
        batch_size, seq_len, _ = q.size()

        # 线性变换
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # 计算注意力分数
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # 应用稀疏掩码
        sparse_mask = self.sparse_mask[:seq_len, :seq_len]
        attn_scores = attn_scores.masked_fill(~sparse_mask, float("-inf"))

        # 计算注意力权重
        attn_weights = F.softmax(attn_scores, dim=-1)

        # 加权求和
        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return output


class TransformerPoem(nn.Module):
    """改进的Transformer模型"""

    def __init__(
        self,
        vocab_size,
        d_model=512,
        n_heads=8,
        num_layers=12,
        max_len=128,
        use_glu=False,
        use_relative_pos=False,
        use_sparse_attn=False,
        residual_before_ln=True,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_len, d_model)
        self.use_relative_pos = use_relative_pos
        if use_relative_pos:
            self.relative_pos = RelativePositionEmbedding(max_len, d_model)
        self.use_sparse_attn = use_sparse_attn

        # 编码器和解码器
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=2048,
            norm_first=not residual_before_ln,  # 控制残差连接的位置
        )
        self.encoder = TransformerEncoder(encoder_layer, num_layers)

        decoder_layer = TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=2048,
            norm_first=not residual_before_ln,  # 控制残差连接的位置
        )
        self.decoder = TransformerDecoder(decoder_layer, num_layers)

        # 输出层
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.use_glu = use_glu
        if use_glu:
            self.glu = GLU(d_model)

        # 稀疏注意力
        if use_sparse_attn:
            self.sparse_attn = SparseAttention(d_model, n_heads)

        self.apply(init_weights)

    def forward(self, src, tgt):
        src_seq_len, tgt_seq_len = src.size(1), tgt.size(1)
        src_pos = torch.arange(src_seq_len, device=src.device).unsqueeze(0)
        tgt_pos = torch.arange(tgt_seq_len, device=tgt.device).unsqueeze(0)

        src_embed = self.embedding(src) + self.pos_embedding(src_pos)
        tgt_embed = self.embedding(tgt) + self.pos_embedding(tgt_pos)

        if self.use_relative_pos:
            # 获取相对位置编码并广播到 [batch_size, seq_len, d_model]
            src_relative_pos = self.relative_pos(src_seq_len, src_embed.device).unsqueeze(0).transpose(1, 2)
            tgt_relative_pos = self.relative_pos(tgt_seq_len, tgt_embed.device).unsqueeze(0).transpose(1, 2)
            # print("src_relative_pos shape:", src_relative_pos.shape)  # ([1, 512, 127])
            # print("src_embed shape:", src_embed.shape)
            src_embed += src_relative_pos
            tgt_embed += tgt_relative_pos

        memory = self.encoder(src_embed)
        output = self.decoder(tgt_embed, memory)

        if self.use_glu:
            output = self.glu(output)

        if self.use_sparse_attn:
            output = self.sparse_attn(output, output, output)

        return self.fc_out(output)


class TransformerEncoderLayerCustom(nn.Module):
    """自定义 Transformer 编码器层"""

    def __init__(self, d_model, n_heads, dim_feedforward=2048, dropout=0.1, use_glu=False, residual_before_ln=True):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.use_glu = use_glu
        self.residual_before_ln = residual_before_ln  # 控制残差连接的位置
        if use_glu:
            self.glu = GLU(d_model)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # 自注意力
        src2 = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        if self.residual_before_ln:
            src = src + self.dropout1(src2)  # 残差连接在前
            src = self.norm1(src)  # LayerNorm 在后
        else:
            src = self.norm1(src + self.dropout1(src2))  # LayerNorm 在前，残差连接在后

        # 前馈网络
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        if self.use_glu:
            src2 = self.glu(src2)
        if self.residual_before_ln:
            src = src + self.dropout2(src2)  # 残差连接在前
            src = self.norm2(src)  # LayerNorm 在后
        else:
            src = self.norm2(src + self.dropout2(src2))  # LayerNorm 在前，残差连接在后

        return src


class TransformerDecoderLayerCustom(nn.Module):
    """自定义 Transformer 解码器层"""

    def __init__(self, d_model, n_heads, dim_feedforward=2048, dropout=0.1, use_glu=False, residual_before_ln=True):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.multihead_attn = MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.use_glu = use_glu
        self.residual_before_ln = residual_before_ln  # 控制残差连接的位置
        if use_glu:
            self.glu = GLU(d_model)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        # 自注意力
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        if self.residual_before_ln:
            tgt = tgt + self.dropout1(tgt2)  # 残差连接在前
            tgt = self.norm1(tgt)  # LayerNorm 在后
        else:
            tgt = self.norm1(tgt + self.dropout1(tgt2))  # LayerNorm 在前，残差连接在后

        # 编码器-解码器注意力
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)[0]
        if self.residual_before_ln:
            tgt = tgt + self.dropout2(tgt2)  # 残差连接在前
            tgt = self.norm2(tgt)  # LayerNorm 在后
        else:
            tgt = self.norm2(tgt + self.dropout2(tgt2))  # LayerNorm 在前，残差连接在后

        # 前馈网络
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        if self.use_glu:
            tgt2 = self.glu(tgt2)
        if self.residual_before_ln:
            tgt = tgt + self.dropout3(tgt2)  # 残差连接在前
            tgt = self.norm3(tgt)  # LayerNorm 在后
        else:
            tgt = self.norm3(tgt + self.dropout3(tgt2))  # LayerNorm 在前，残差连接在后

        return tgt


class TransformerPoemCustom(nn.Module):
    """使用 PyTorch 基础模块实现的 Transformer 模型"""

    def __init__(
        self,
        vocab_size,
        d_model=512,
        n_heads=8,
        num_layers=6,
        max_len=128,
        use_glu=False,
        use_relative_pos=False,
        use_sparse_attn=False,
        residual_before_ln=True,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_len, d_model)
        self.use_relative_pos = use_relative_pos
        if use_relative_pos:
            self.relative_pos = RelativePositionEmbedding(max_len, d_model)
        self.use_sparse_attn = use_sparse_attn

        # 编码器和解码器
        self.encoder = nn.ModuleList(
            [
                TransformerEncoderLayerCustom(d_model, n_heads, use_glu=use_glu, residual_before_ln=residual_before_ln)
                for _ in range(num_layers)
            ]
        )
        self.decoder = nn.ModuleList(
            [
                TransformerDecoderLayerCustom(d_model, n_heads, use_glu=use_glu, residual_before_ln=residual_before_ln)
                for _ in range(num_layers)
            ]
        )

        # 输出层
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt):
        src_seq_len, tgt_seq_len = src.size(1), tgt.size(1)
        src_pos = torch.arange(src_seq_len, device=src.device).unsqueeze(0)
        tgt_pos = torch.arange(tgt_seq_len, device=tgt.device).unsqueeze(0)

        src_embed = self.embedding(src) + self.pos_embedding(src_pos)
        tgt_embed = self.embedding(tgt) + self.pos_embedding(tgt_pos)

        if self.use_relative_pos:
            src_embed += self.relative_pos(src_seq_len)
            tgt_embed += self.relative_pos(tgt_seq_len)

        # 编码器
        memory = src_embed
        for layer in self.encoder:
            memory = layer(memory)

        # 解码器
        output = tgt_embed
        for layer in self.decoder:
            output = layer(output, memory)

        return self.fc_out(output)


# 测试模型
if __name__ == "__main__":
    model = TransformerPoem(vocab_size=10000, use_sparse_attn=True)
    src = torch.randint(0, 10000, (32, 128))
    tgt = torch.randint(0, 10000, (32, 128))
    output = model(src, tgt)
    print(output.shape)
