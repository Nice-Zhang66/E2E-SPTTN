from typing import Optional
import numpy as np
import torch
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import nn
from torch.autograd import Variable
# from featuremap_visual import featuremap_visual
from model.TransformerLayer import TransformerEncoder, TransformerDecoder
from model.TransformerLayer import PositionalEncoding


def tgt_subsequent_mask(size):
    """Mask out subsequent positions."""
    attn_shape = (1, size, size)
    tgt_subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(tgt_subsequent_mask) == 0


def make_std_mask(tgt, pad=0):
    """Create a mask to hide padding and future words."""
    tgt_mask = (tgt != pad).unsqueeze(-2)
    tgt_mask = tgt_mask & generate_square_subsequent_mask(tgt.size(-1)).type_as(tgt_mask)
    # tgt.size(-1) 表示的是序列的长度
    return tgt_mask


def sequence_mask(seq):
    batch_size, seq_len = seq.size()
    mask = 1 - torch.triu(torch.ones((seq_len, seq_len), dtype=torch.uint8),diagonal=1)
    mask = mask.unsqueeze(0).expand(batch_size, -1, -1)  # [B, L, L]
    return mask

def get_pad_mask(seq, pad_idx=0):
    return (seq != pad_idx)


def subsequent_mask(tensor, tgt):
    seq, batch_size, _ = tgt.size()
    batch_size, seq_len, _ = tensor.size()
    mask = (torch.triu(torch.ones(seq_len, seq)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def generate_square_subsequent_mask(sz: int):
    r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
    """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


class Transformer(nn.Module):

    def __init__(self, image_size, vocab_size, patch_size=16, channels=3,
                 embed_dim=512, d_model=512, num_heads=4, num_encoder_layers=4, num_decoder_layers=4,
                 dim_feedforward=2048, dropout=0.1):
        super(Transformer, self).__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        # network architecture
        self.Embedding = nn.Embedding(vocab_size, d_model)
        self.patch_dim = channels * patch_size ** 2
        self.to_patch_embedding = nn.Sequential(Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                                                          p1=patch_size, p2=patch_size),
                                                nn.Linear(self.patch_dim, embed_dim))
        # Positional Embeddings
        self.pe = PositionalEncoding(d_model, dropout,vocab_size)
        self.pos_drop = nn.Dropout(p=dropout)
        self._reset_parameters()
        # self.linear = nn.Linear(1024, d_model)
        self.encoder = TransformerEncoder(d_model, num_heads, num_encoder_layers,
                                          dim_feedforward, dropout=0.1, norm=None)
        self.decoder = TransformerDecoder(num_decoder_layers, num_heads, embed_dim, dim_feedforward,
                                          dropout, emb_dropout=dropout, vocab_size=vocab_size)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, tgt):
        b, t, c, w, h = src.shape

        # hide time inside batch
        # x = src.permute(0, 2, 1, 3, 4)  # (b, t, c, h, w)
        x = rearrange(src, 'b t c h w -> (b t) c h w')  # (b*t, c, h, w)
        # 特征可视化
        # featuremap_visual(x, "patch之前")
        x = self.to_patch_embedding(x)  # (b*t, n, d)

        src = rearrange(x, '(b t) n d -> b t n d', b=b, t=t)  # (b, t, n, d)
        # featuremap_visual(src, "patch之后")
        # src_mask = get_pad_mask(src, 0)
        # Create a matrix mask to avoid seeing future words in tgt
        # Same mask applied to all h heads.
        tgt_pad_mask = get_pad_mask(tgt, 0)
        attn_mask = generate_square_subsequent_mask(tgt.size(-1))
        b, t, n, d = src.shape

        pos_embedding = nn.Parameter(torch.randn(1, t, n, d))
        src += pos_embedding.cuda()  # (b, t, n, d)
        src = self.pos_drop(src)
        # featuremap_visual(src, "加上pos")
        # tgt_sequent_mask = generate_square_subsequent_mask(tgt.size(1))
        # attn_mask = tgt_mask & tgt_sequent_mask
        tgt = self.pe(self.Embedding(tgt))
        tgt = tgt.permute(1, 0, 2)
        memory = self.encoder(src=src)
        memory_mask = subsequent_mask(memory, tgt)
        memory = memory.permute(1, 0, 2)
        outputs = self.decoder(tgt, memory, memory_mask.cuda(), attn_mask.cuda(), tgt_pad_mask.cuda())
        return outputs
