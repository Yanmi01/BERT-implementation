import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class Config:
    def __init__(self,
                 vocab_size=30522,
                 hidden_size=768,  # Embedding size
                 num_hidden_layers=12,  # Number of transformer layers
                 num_attention_heads=12,  # Number of attention heads
                 intermediate_size=3072,  # Size of the intermediate layer in the MLP
                 dropout_prob=0.1,  # Dropout probability
                 max_position_embeddings=512,  # Maximum sequence length
                 type_vocab_size=2,  # Number of segment types
                 initializer_range=0.02):  # Range for weight initialization

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = dropout_prob
        self.attention_probs_dropout_prob = dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range

    @classmethod
    def from_dict(cls, config_dict):
        """Create a Config instance from a dictionary."""
        return cls(**config_dict)


class MLP(nn.Module):
    """A simple feed-forward neural network with GELU activation."""
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Linear(intermediate_size, hidden_size)
        )

    def forward(self, x):
        return self.mlp(x)


class Layer(nn.Module):
    """A single transformer layer consisting of self-attention and MLP."""
    def __init__(self, config):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # Self-attention layers
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.attn_out = nn.Linear(config.hidden_size, config.hidden_size)

        # Layer normalization
        self.ln1 = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.ln2 = nn.LayerNorm(config.hidden_size, eps=1e-12)

        # MLP
        self.mlp = MLP(config.hidden_size, config.intermediate_size)

    def split_heads(self, tensor, num_heads, attention_head_size):
        """Split the hidden size into multiple attention heads."""
        new_shape = tensor.size()[:-1] + (num_heads, attention_head_size)
        return tensor.view(*new_shape).permute(0, 2, 1, 3)

    def merge_heads(self, tensor, num_heads, attention_head_size):
        """Merge multiple attention heads back into the hidden size."""
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attention_head_size,)
        return tensor.view(new_shape)

    def attn(self, q, k, v, attention_mask):
        """Compute the self-attention mechanism."""
        mask = attention_mask == 1
        mask = mask.unsqueeze(1).unsqueeze(2)

        # Compute attention scores
        s = torch.matmul(q, k.transpose(-1, -2))
        s = s / math.sqrt(self.attention_head_size)

        # Apply attention mask
        s = s.masked_fill_(mask, float('-inf'))
        # s = torch.where(mask, s, torch.tensor(float('-inf'))) you can also choose this line


        # Compute attention probabilities
        p = F.softmax(s, dim=-1)
        p = self.dropout(p)

        # Compute attention output
        a = torch.matmul(p, v)
        return a

    def forward(self, x, attention_mask):
        """Forward pass for the transformer layer."""
        # Save the input for the residual connection
        res = x

        # Self-attention block
        q, k, v = self.query(x), self.key(x), self.value(x)
        q = self.split_heads(q, self.num_attention_heads, self.attention_head_size)
        k = self.split_heads(k, self.num_attention_heads, self.attention_head_size)
        v = self.split_heads(v, self.num_attention_heads, self.attention_head_size)

        a = self.attn(q, k, v, attention_mask)
        a = self.merge_heads(a, self.num_attention_heads, self.attention_head_size)
        a = self.attn_out(a)
        a = self.dropout(a)
        a = self.ln1(a + res)  # Residual connection + layer norm

        # MLP block
        m = self.mlp(a)
        m = self.dropout(m)
        m = self.ln2(m + a)  # Residual connection + layer norm

        return m


class Bert(nn.Module):
    """The BERT model."""
    def __init__(self, config_dict):
        super().__init__()
        self.config = Config.from_dict(config_dict)

        # Embeddings
        self.embeddings = nn.ModuleDict({
            'token': nn.Embedding(self.config.vocab_size, self.config.hidden_size, padding_idx=0),
            'position': nn.Embedding(self.config.max_position_embeddings, self.config.hidden_size),
            'token_type': nn.Embedding(self.config.type_vocab_size, self.config.hidden_size),
        })

        # Layer normalization and dropout
        self.ln = nn.LayerNorm(self.config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)

        # Transformer layers
        self.layers = nn.ModuleList([
            Layer(self.config) for _ in range(self.config.num_hidden_layers)
        ])

        # Pooler
        self.pooler = nn.Sequential(OrderedDict([
            ('dense', nn.Linear(self.config.hidden_size, self.config.hidden_size)),
            ('activation', nn.Tanh()),
        ]))

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        """Forward pass for the BERT model."""
        # Generate position IDs
        position_ids = torch.arange(input_ids.size(1), dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        # Use default token type IDs if not provided
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # Sum the embeddings
        x = self.embeddings.token(input_ids) + self.embeddings.position(position_ids) + self.embeddings.token_type(token_type_ids)
        x = self.dropout(self.ln(x))

        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x, attention_mask)

        # Pool the output
        pooled_output = self.pooler(x[:, 0])
        return x, pooled_output

    def load_model(self, path):
        """Load model weights from a file."""
        self.load_state_dict(torch.load(path))
        return self