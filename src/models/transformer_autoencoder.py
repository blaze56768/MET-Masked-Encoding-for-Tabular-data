import torch
import torch.nn as nn
import lightning as L
from models.modules.autoencoders.transformer import masking, positional_embedding, encoder, decoder
from torch import optim

class TransformerAutoEncoder(L.LightningModule):
    """Masked Transformer Model for Self-Supervised Learning (SSL).

    This model uses masking strategies and transformer blocks for self-supervised learning,
    and includes methods for training and testing.

    Args:
        cfg (object): Configuration object containing model parameters.
    """
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(logger=True)
        self.cfg = cfg
  
        self.masker = masking.Masker(self.cfg)
        self.position_embedding = positional_embedding.PositionalEmbedding(self.cfg)
        self.encoder = encoder.TransformerEncoder(self.cfg)
        self.decoder = decoder.TransformerDecoder(self.cfg)
        
        self.projection_head = nn.Linear(in_features=self.cfg.model_params.pos_embedding_dim + 1, out_features=self.cfg.model_params.output_features)

        self.mse_loss = torch.nn.functional.mse_loss
        self.bce_loss = torch.nn.functional.binary_cross_entropy_with_logits

    
    def add_pos_embeddings(self,x, pos_emb):
        """Apply masking to the input data."""

        x = torch.unsqueeze(x, -1).float()
        x_emb = torch.cat((x, pos_emb), dim=2)
        
        return x_emb

    def forward(self, inputs, masking_pct):
        """Forward pass for the Masked Transformer Model."""

        value_mask, idx_mask, value_unmask, idx_unmask = self.masker.forward(inputs, mask_pct=masking_pct)
        
        #get positional embeddings
        pos_emb_mask = self.position_embedding.forward(idx_mask)
        pos_emb_unmask = self.position_embedding.forward(idx_unmask)

        input_mask = self.add_pos_embeddings(value_mask, pos_emb_mask)
        input_unmask = self.add_pos_embeddings(value_unmask, pos_emb_unmask)

        output_encoder = self.encoder.forward(input_unmask)
        input_decoder = torch.cat((output_encoder, input_mask), dim=1).float()
        
        output_decoder = self.decoder.forward(input_decoder)
        
        
        return output_decoder
    

    def get_embeddings(self, inputs, masking_pct):
        """Get embeddings for downstream tasks."""
        embeddings, _ = self.forward(inputs, masking_pct)
        return embeddings

    def training_step(self, batch, batch_idx):
        """Training step for the Masked Transformer Model."""
        # inputs = batch[:, :-2]
        # targets = batch[:, :-2]
        inputs, labels = batch
       
        output_decoder = self(inputs, self.cfg.model_params.masking_pct)
        projected = self.projection_head(output_decoder)
        outputs = torch.squeeze(projected, dim=2)
        loss = self.mse_loss(outputs, inputs)
        logs = {"train/loss": loss}
        self.log_dict(logs, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        
        return {"loss": loss}
    
    def validation_step(self, batch, batch_idx):
        """Validation step for the Masked Transformer Model."""
        # inputs = batch[:, :-1]
        # targets = batch[:, :-1]
        inputs, labels = batch
       
        output_decoder = self(inputs, self.cfg.model_params.masking_pct)
        projected = self.projection_head(output_decoder)
        outputs = torch.squeeze(projected, dim=2)
        loss = self.mse_loss(outputs, inputs)
    
        logs = {"val/loss": loss}
        self.log_dict(logs, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        return {"loss": loss}
    
    def test_step(self, batch, batch_idx):
        """Test step for the Masked Transformer Model."""
        # inputs = batch[:, :-1]
        # targets = batch[:, :-1]
        inputs, labels = batch
       
        output_decoder = self(inputs, self.cfg.model_params.masking_pct)
        projected = self.projection_head(output_decoder)
        outputs = torch.squeeze(projected, dim=2)
        loss = self.mse_loss(outputs, inputs)
        logs = {"test/loss": loss}
        self.log_dict(logs, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        return {"loss": loss}

    def reconstruction_loss(self, outputs, targets):
        """Calculates the reconstruction loss.

        Args:
            outputs (torch.Tensor): Model outputs.
            targets (torch.Tensor): Target values.

        Returns:
            tuple: Numerical loss and categorical loss.
        """
        outputs_categorical, outputs_numerical = self.split_tensor(outputs)
        targets_categorical, targets_numerical = self.split_tensor(targets)

        loss_num = self.mse_loss(outputs_numerical, targets_numerical)
        loss_cat = self.bce_loss(outputs_categorical, targets_categorical)
        return loss_num, loss_cat
        
    def split_tensor(self, tensor):
        """Splits the tensor into categorical and numerical features.

        Args:
            tensor (torch.Tensor): Input tensor.

        Returns:
            tuple: Split tensors for categorical and numerical features.
        """
        return torch.split(tensor, [self.cfg.model_params.num_categorical_features, self.cfg.model_params.num_numerical_features], dim=1)
    
    def train_start(self):
        """Callback for actions to take when training starts."""
        print("✅ Training is about to start")
  
    def train_end(self):
        """Callback for actions to take when training ends."""
        print("🎉 Training has ended 🎉")
        
    def test_start(self):
        """Callback for actions to take when testing starts."""
        print("✅ Testing is about to start")
  
    def test_end(self):
        """Callback for actions to take when testing ends."""
        print("🎉 Testing has ended 🎉")

    def configure_optimizers(self):
        """Configures the optimizers and learning rate scheduler.

        Returns:
            tuple: Optimizer and learning rate scheduler.
        """
        optimizer = optim.Adam(params=self.parameters(), lr=self.cfg.train_params.lr, amsgrad=True)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, self.cfg.train_params.lr_step_size)
        return [optimizer], [scheduler]
