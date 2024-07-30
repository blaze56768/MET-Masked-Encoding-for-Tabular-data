from torch import nn


class PositionalEmbedding(nn.Module):
    """Computes positional embeddings for both masked and unmasked positions.

    This module generates an array containing positional embeddings.

    Args:
        cfg (object): Configuration object containing model parameters.
    """
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embedding_dim = self.cfg.model_params.pos_embedding_dim
        self.position_embedding = nn.Embedding(num_embeddings=self.cfg.model_params.num_features, embedding_dim=self.cfg.model_params.pos_embedding_dim)
      
    def forward(self, indices):
        """Generates positional embeddings.

        Args:
            indices (torch.Tensor): Indices of elements to embed.

        Returns:
            torch.Tensor: Positional embeddings for the given indices.
        """
        position_embedding = self.position_embedding(indices)
        return position_embedding

        
        
 
        
        
        
        
        