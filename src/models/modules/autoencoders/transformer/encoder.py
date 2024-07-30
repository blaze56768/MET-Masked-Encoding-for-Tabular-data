import torch.nn as nn
from models.modules.autoencoders.transformer import transformer_block


class TransformerEncoder(nn.Module):
    """Transformer Encoder for Self-Supervised Learning.

    This module applies positional embedding and multiple transformer blocks for encoding.

    Args:
        cfg (object): Configuration object containing model parameters.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.encoder_blocks = nn.ModuleList([transformer_block.TransformerBlock(self.cfg) for _ in range(self.cfg.model_params.num_encoder_blocks)])

    def forward(self, x):
        """Forward pass for the Transformer Encoder.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Encoded output.
        """

        for block in self.encoder_blocks:
            x = block(x)

        return x