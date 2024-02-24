import collections.abc as container_abcs
import math
from itertools import repeat
from typing import Optional
# from featuremap_visual import featuremap_visual

import torch
from einops import rearrange
from torch import nn
from torch.autograd import Variable
from torch.functional import Tensor, einsum
import torch.nn.functional as F


class TransformerEncoder(nn.Module):

    def __init__(self, d_model, num_heads, num_layers, dim_feedforward=2048, dropout=0.1, norm=None):
        super(TransformerEncoder,self).__init__()
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(d_model, num_heads, dim_feedforward, dropout)
                for _ in range(num_layers)
            ]
        )
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                ):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask, pos=pos
                           )

        if self.norm is not None:
            output = self.norm(output)

        output = rearrange(output,'b t n d -> b (t n) d')
        return output

    def __repr__(self):
        return "%s(num_layers=%r, num_heads=%r)" % (
            self.__class__.__name__,
            len(self.layers),
            self.layers[0],
        )


class TransformerDecoder(nn.Module):
    """
    A transformer decoder with N masked layers.
    Decoder layers are masked so that an attention head cannot see the future.
    """

    def __init__(
        self,
        num_layers, num_heads, hidden_size, ff_size: int = 2048,
        dropout: float = 0.1, emb_dropout: float = 0.1,
        vocab_size: int = 10000,
        **kwargs
    ):

        super(TransformerDecoder, self).__init__()

        self._output_size = vocab_size

        # create num_layers decoder layers and put them in a list
        self.layers = nn.ModuleList(
            [
                TransformerDecoderLayer(d_model=512, nhead=num_heads, dim_feedforward=ff_size, dropout=dropout)
                for _ in range(num_layers)
            ]
        )

        self.pe = PositionalEncoding(hidden_size, dropout, vocab_size)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-6)

        self.emb_dropout = nn.Dropout(p=emb_dropout)
        self.output_layer = nn.Linear(hidden_size, self._output_size, bias=False)


    def forward(
        self,
        trg_embed: Tensor = None,
        encoder_output: Tensor = None,
        src_mask: Tensor = None,
        attn_mask: Tensor = None,
        tgt_pad_mask: Tensor = None,
        **kwargs
    ):
        global y
        assert attn_mask is not None, "trg_mask required for Transformer"

        x = self.pe(trg_embed)  # add position encoding to word embedding
        x = self.emb_dropout(x)

        for layer in self.layers:
            y = layer(tgt=x, memory=encoder_output, memory_mask=src_mask, attn_mask=attn_mask)
        output = self.layer_norm(y)
        output = self.output_layer(output)

        return output


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, num_heads, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        heads_half = int(num_heads / 2.0)
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        self.attention_space = PreNorm(d_model, Attention(d_model, heads=heads_half, dim_head=128, dropout=dropout,
                                       attn_type='space'))
        self.attention_time = PreNorm(d_model, Attention(d_model, heads=heads_half, dim_head=128, dropout=dropout,
                                      attn_type='time'))
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.feed_forward = PositionwiseFeedForward(
            input_size=d_model, ff_size=dim_feedforward, dropout=dropout
        )
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, src, src_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        # src2 = self.self_attn(q, k, value=src, attn_mask=src_mask)[0]
        b, t, n, d = src.shape
        xs = self.attention_space(src)
        xt = self.attention_time(src)
        src2 = torch.cat([xs, xt], dim=3)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src = self.feed_forward(src)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, pos)
        return self.forward_post(src, src_mask, pos)


class TransformerDecoderLayer(nn.Module):
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu"):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(
            input_size=d_model, ff_size=dim_feedforward, dropout=dropout
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        # self.dropout3 = nn.Dropout(dropout)
        self.heads = nhead
        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory, memory_mask: Optional[Tensor] = None,
                attn_mask: Optional[Tensor] = None, tgt_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None, query_pos: Optional[Tensor] = None):
        tgt_norm = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt_norm, query_pos)
        # B, L, S = attn_mask.shape
        # # torch.stack([attn_mask for _ in range(self.heads)], dim=1).contiguous().view(-1, L, S)
        # attn_masks = attn_mask.repeat(1, self.heads, 1).contiguous().view(B * self.heads, L, S)
        tgt2 = self.self_attn(q, k, value=tgt_norm, key_padding_mask=tgt_padding_mask, attn_mask=attn_mask)[0]
        tgt = tgt + self.dropout1(tgt2)

        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask)[0]
        tgt3 = tgt + self.dropout2(tgt2)
        tgt = self.feed_forward(tgt3)
        # tgt = self.norm3(tgt + tgt3) # 生成式的
        return tgt


class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-forward layer
    Projects to ff_size and then back down to input_size.
    """

    def __init__(self, input_size, ff_size, dropout=0.1):
        """
        Initializes position-wise feed-forward layer.
        :param input_size: dimensionality of the input.
        :param ff_size: dimensionality of intermediate representation
        :param dropout:
        """
        super(PositionwiseFeedForward, self).__init__()
        self.layer_norm = nn.LayerNorm(input_size, eps=1e-6)
        self.pwff_layer = nn.Sequential(
            nn.Linear(input_size, ff_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_size, input_size),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x_norm = self.layer_norm(x)
        return self.pwff_layer(x_norm) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.5, attn_type=None):
        super().__init__()

        assert attn_type in ['space', 'time'], 'Attention type should be one of the following: space, time.'

        self.attn_type = attn_type

        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

    def forward(self, x):
        # reshape to reveal dimensions of space and time
        if self.attn_type == 'space':
            out = self.forward_space(x)  # (b, t, n, d)
        elif self.attn_type == 'time':
            out = self.forward_time(x)  # (b, t, n, d)
        else:
            raise Exception('Unknown attention type: %s' % (self.attn_type))

        return out

    def forward_space(self, x):
        """
        x: (b, t, n, d)
        """
        b, t, n, d = x.shape

        # hide time dimension into batch dimension
        x = rearrange(x, 'b t n d -> (b t) n d')  # (bt, n, d)

        # apply self-attention
        out = self.forward_attention(x)  # (bt, n, d)

        # recover time dimension and merge it into space
        out = rearrange(out, '(b t) n d -> b t n d', t=t, n=n)  # (b, tn, d)

        return out

    def forward_time(self, x):
        """
        x: (b, t, n, d)
        """
        b, t, n, d = x.shape

        # t = self.num_patches_time
        # n = self.num_patches_space

        # hide time dimension into batch dimension
        x = rearrange(x, 'b t n d -> (b n) t d')  # (bn, t, d)

        # apply self-attention
        out = self.forward_attention(x)  # (bn, t, d)

        # recover time dimension and merge it into space
        out = rearrange(out, '(b n) t d -> b t n d', t=t, n=n)  # (b, tn, d)

        return out

    def forward_attention(self, x):
        h = self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        # q = q * self.scale  #这一句是干什么的

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return out



def _ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse

to_2tuple = _ntuple(2)


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=512):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches  # 196
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, T, H, W = x.shape
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.proj(x)
        W = x.size(-1)
        x = x.flatten(2).transpose(1, 2)
        return x, T, W

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len):
        """
        位置编码器类的初始化函数

        共有三个参数，分别是
        d_model：词嵌入维度
        dropout: dropout触发比率
        max_len：每个句子的最大长度
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings
        # 注意下面代码的计算方式与公式中给出的是不同的，但是是等价的，你可以尝试简单推导证明一下。
        # 这样计算是为了避免中间的数值计算结果超出float的范围，
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        pos = Variable(self.pe[:, :x.size(1)], requires_grad=False)
        # x = x + self.pe[:, :x.size(1)].requires_grad_(False)
        return self.dropout(x + pos)


