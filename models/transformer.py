import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, ff_dim=1024, dropout=0.1):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim))

        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, attention_mask=None):
        attn_output, _ = self.attention(x, x, x,
            attn_mask=attention_mask,
            need_weights=False
        )
        x = self.norm1(x + self.dropout1(attn_output))

        ff_output = self.feed_forward(x)

        x = self.norm2(x + self.dropout2(ff_output))

        return x


class VeloraTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim=256,
        num_layers=4,
        num_heads=8,
        ff_dim=1024,
        block_size=128,
        dropout=0.1):

        super().__init__()

        self.block_size = block_size

        self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.position_embedding = nn.Embedding(block_size, embed_dim)

        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    ff_dim=ff_dim,
                    dropout=dropout
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

        self.output = nn.Linear(embed_dim, vocab_size)

        causal_mask = torch.triu(
            torch.ones(block_size, block_size, dtype=torch.bool),
            diagonal=1
        )
        self.register_buffer("causal_mask", causal_mask)


    def forward(self, x):
        batch_size, seq_length = x.shape

        if seq_length > self.block_size:
            raise ValueError(f"Sequence length {seq_length} exceeds block_size {self.block_size}")

        positions = torch.arange(seq_length, device=x.device).unsqueeze(0)

        x = self.embedding(x) + self.position_embedding(positions)
        x = self.dropout(x)

        attention_mask = self.causal_mask[:seq_length, :seq_length]

        for layer in self.layers:
            x = layer(x, attention_mask=attention_mask)

        x = self.norm(x)

        logits = self.output(x)

        return logits
