

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super().__init__()

        assert embedding_dim % num_heads == 0

        self.head_dim = embedding_dim // num_heads
        self.num_heads = num_heads

        self.query = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.key = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.value = nn.Linear(embedding_dim, embedding_dim, bias=False)

        self.output_projection = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x):
        B, T, C = x.shape

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        Q = Q.reshape(B, T, self.num_heads, self.head_dim)
        K = K.reshape(B, T, self.num_heads, self.head_dim)
        V = V.reshape(B, T, self.num_heads, self.head_dim)

        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        scores = Q @ K.transpose(-2, -1)
        scores = scores / math.sqrt(self.head_dim)

        mask = torch.tril(
            torch.ones(T, T, device=x.device, dtype=torch.bool)
        )

        scores = scores.masked_fill(~mask, float("-inf"))

        weights = F.softmax(scores, dim=-1)

        output = weights @ V

        output = output.transpose(1, 2).contiguous()
        output = output.reshape(B, T, C)

        output = self.output_projection(output)

        return output
class FeedForward(nn.Module):
  def __init__(self, embedding_dim, hidden_dim):
    super().__init__()

    self.linear1 = nn.Linear(embedding_dim, hidden_dim)
    self.linear2 = nn.Linear(hidden_dim, embedding_dim)

  def forward(self, x):
    x = self.linear1(x)
    x = F.relu(x)
    x = self.linear2(x)

    return x
class TransformerBlock(nn.Module):
  def __init__(self, embedding_dim, num_heads, hidden_dim):
    super().__init__()
    self.attention = MultiHeadAttention(embedding_dim, num_heads)
    self.ffn = FeedForward(embedding_dim, hidden_dim)

    self.ln1 = nn.LayerNorm(embedding_dim)
    self.ln2 = nn.LayerNorm(embedding_dim)

  def forward(self, x):
    x = x + self.attention(self.ln1(x))
    x = x + self.ffn(self.ln2(x))

    return x
class GPTLanguageModel(nn.Module):
  def __init__(self, vocab_size, embedding_dim=64, num_heads=4, hidden_dim=256, num_layers=2, block_size=128):
    super().__init__()

    self.block_size = block_size
    self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
    self.position_embedding = nn.Embedding(block_size, embedding_dim)

    self.blocks = nn.Sequential(
        *[
          TransformerBlock(embedding_dim, num_heads, hidden_dim)
          for _ in range(num_layers)
          ]
    )

    self.final_ln = nn.LayerNorm(embedding_dim)
    self.lm_head = nn.Linear(embedding_dim, vocab_size)

  def forward(self, x):
    B, T = x.shape

    positions = torch.arange(T, device=x.device)

    token_embeddings = self.token_embedding(x)
    position_embeddings = self.position_embedding(positions)

    x = token_embeddings + position_embeddings

    x = self.blocks(x)
    x = self.final_ln(x)

    logits = self.lm_head(x)

    return logits

