import torch.nn as nn
from models.modules.autoencoders.transformer import transformer_block


class TransformerDecoder(nn.Module):
    """Transformer Decoder for Self-Supervised Learning.

    This module applies multiple transformer blocks for decoding.

    Args:
        cfg (object): Configuration object containing model parameters.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.decoder_blocks = nn.ModuleList([transformer_block.TransformerBlock(self.cfg) for _ in range(self.cfg.model_params.num_decoder_blocks)])

    def forward(self, x):
        """Forward pass for the Transformer Decoder.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Decoded output.
        """
        
        for block in self.decoder_blocks:
            x = block(x)

        return x