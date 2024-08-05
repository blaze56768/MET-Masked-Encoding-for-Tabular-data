import numpy as np
import torch

rng = np.random.default_rng()

class Masker(torch.nn.Module):
    """This Module masks the inputs based on the masking mode and percentage.

    Args:
        cfg (object): Configuration object containing device settings.
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def forward(self, x, mask_pct=0.5):
        """Forward pass for masking input data.

        Args:
            x (torch.Tensor): Input data tensor.
            mask_pct (float): Percentage of data to mask.

        Returns:
            tuple: Masked data tensor, indices of masked elements, 
                   unmasked data tensor, indices of unmasked elements.
        """
        masked_data, masked_indices, unmasked_data, unmasked_indices = self.zero_masking(x, mask_pct)
        
        return masked_data, masked_indices, unmasked_data, unmasked_indices

    def zero_masking(self, x, mask_pct):
        """Masks random positions in the input tensor with zeros.

        Args:
            x (torch.Tensor): Input data tensor.
            mask_pct (float): Percentage of data to mask.

        Returns:
            tuple: Masked data tensor, indices of masked elements, 
                   unmasked data tensor, indices of unmasked elements.
        """
        rows, cols = x.shape
        x = x.to(self.cfg.train_params.device)

        num_masked = int(cols * mask_pct)
        num_unmasked = cols - num_masked

        shuffled_indices = torch.stack([torch.randperm(cols) for _ in range(rows)]).to(self.cfg.train_params.device)
        masked_indices = shuffled_indices[:, :num_masked]
        unmasked_indices = shuffled_indices[:, num_masked:]

        masked_indices, _ = torch.sort(masked_indices, dim=1)
        unmasked_indices, _ = torch.sort(unmasked_indices, dim=1)

        unmasked_data = torch.zeros((rows, num_unmasked), device=self.cfg.train_params.device)
        masked_data = torch.zeros((rows, num_masked), device=self.cfg.train_params.device)

        row_indices = torch.arange(rows).unsqueeze(1).expand(-1, num_unmasked)
        unmasked_data = x[row_indices, unmasked_indices]

        return (
            masked_data,
            masked_indices,
            unmasked_data,
            unmasked_indices
        )





        # shuffled_indices = np.array([rng.permutation(cols) for _ in range(rows)])
        # masked_indices = shuffled_indices[:, :num_masked]
        # unmasked_indices = shuffled_indices[:, num_masked:]

        # masked_indices.sort(axis=1)
        # unmasked_indices.sort(axis=1)

        # unmasked_data = np.zeros((rows, num_unmasked))
        # masked_data = np.zeros((rows, num_masked))

        # for i in range(rows):
        #     unmasked_data[i] = x[i][unmasked_indices[i]]

        # return (
        #     torch.tensor(masked_data).to(self.cfg.train_params.device),
        #     torch.tensor(masked_indices).to(self.cfg.train_params.device),
        #     torch.tensor(unmasked_data).to(self.cfg.train_params.device),
        #     torch.tensor(unmasked_indices).to(self.cfg.train_params.device)
        # )

        