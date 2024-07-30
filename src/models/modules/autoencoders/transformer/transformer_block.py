from torch import nn, relu

class TransformerBlock(nn.Module):
    """Implements a Transformer block with Multi-Head Attention and FeedForward layers.

    Args:
        cfg (object): Configuration object containing model parameters.
    """
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=self.cfg.model_params.pos_embedding_dim + 1,
            num_heads=self.cfg.model_params.num_heads,
            dropout=self.cfg.model_params.dropout,
            batch_first=True
        )
        
        self.feedforward = PositionwiseFeedForward(
            embed_dim=self.cfg.model_params.pos_embedding_dim + 1, 
            feedforward_dim=self.cfg.model_params.feedforward_dim,
            dropout=self.cfg.model_params.dropout
        )

        self.dropout = nn.Dropout(self.cfg.model_params.dropout)
        self.layernorm_attention = nn.LayerNorm(self.cfg.model_params.pos_embedding_dim + 1)
        self.layernorm_feedforward = nn.LayerNorm(self.cfg.model_params.pos_embedding_dim + 1)

    def forward(self, x):
        """Forward pass for the Transformer block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the Transformer block.
        """
        # Multi-Head Attention
        attention_out, _ = self.multihead_attention(x, x, x)
        # Add & Norm
        x = self.layernorm_attention(x + self.dropout(attention_out))
        # Position-wise FeedForward
        feedforward_out = self.feedforward(x)
        # Add & Norm
        x = self.layernorm_feedforward(x + self.dropout(feedforward_out))
        return x

class PositionwiseFeedForward(nn.Module):
    """Implements the Position-wise FeedForward Network (FFN) equation.

    Args:
        embed_dim (int): Dimension of the embedding.
        feedforward_dim (int): Dimension of the feedforward layer.
        dropout (float): Dropout rate.
    """
    
    def __init__(self, embed_dim, feedforward_dim, dropout=0.1):
        super().__init__()
        self.linear_in = nn.Linear(embed_dim, feedforward_dim)
        self.linear_out = nn.Linear(feedforward_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """Forward pass for the Position-wise FeedForward Network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the FeedForward network.
        """
        return self.linear_out(self.dropout(relu(self.linear_in(x))))
