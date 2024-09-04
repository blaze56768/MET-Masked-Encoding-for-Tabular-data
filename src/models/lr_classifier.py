import lightning as L
import torch
import torch.nn as nn
import torchmetrics
import torchmetrics.classification
from src.models import transformer_autoencoder

class LogisticRegressionClassifier(L.LightningModule):
    """Downstream Classifier for leveraging a pretrained model.

    This classifier uses a pretrained self-supervised learning (SSL) model 
    to extract features and perform classification on downstream tasks.

    Args:
        cfg (object): Configuration object containing model parameters.
    """
    
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()

        self.cfg = cfg

        if self.cfg.train_params.downstream_pretrain:
            # Initialize the pretrained LightningModule
            self.pretrained = transformer_autoencoder_old.MaskedTransformerModel.load_from_checkpoint(self.cfg.pretrain_checkpoint)
            # Freeze pretrained encoder
            self.pretrained.freeze()

        # Initialize layers
        self.projection_layer = nn.Linear(self.cfg.model_params.num_features, self.cfg.model_params.output_features)

        # Loss criterion
        self.criterion = nn.BCELoss()

    def forward(self, x):
        """Forward pass for the downstream classifier.

        Args:
            x (torch.Tensor): Input data.

        Returns:
            torch.Tensor: Predictions after applying the projection layer and sigmoid activation.
        """
        if self.cfg.train_params.downstream_pretrain:
            embeddings = self.pretrained(x, 0)
        else:
            embeddings = x
        
        predictions = self.projection_layer(embeddings)
        predictions = torch.sigmoid(predictions)
        predictions = torch.squeeze(predictions, -1)
        
        return predictions

    def training_step(self, batch, batch_idx):
        """Training step for the downstream classifier.

        Args:
            batch (torch.Tensor): Input batch of data.
            batch_idx (int): Index of the batch.

        Returns:
            dict: Dictionary containing the loss.
        """
        inputs = batch[:, :-2]
        labels = batch[:, -1]
        predictions = self(inputs)
        loss = self.criterion(predictions, labels)
        
        acc = torchmetrics.functional.accuracy(predictions, labels.int(), task="binary")
        prec = torchmetrics.functional.precision(predictions, labels, task="binary")
        recall = torchmetrics.functional.recall(predictions, labels.int(), task="binary")
        f1 = torchmetrics.functional.f1_score(predictions, labels.int(), task="binary")
        auroc = torchmetrics.AUROC(task="binary")
        auc = auroc(predictions, labels.int())
        apre = torchmetrics.AveragePrecision(task="binary")
        ap = apre(predictions, labels.int())
        
        logs = {
            "train/loss": loss, "train/accuracy": acc,
            "train/precision": prec, "train/recall": recall, 
            "train/f1": f1, "train/auc": auc,
            "train/ap": ap
        }
        self.log_dict(logs, on_step=True, on_epoch=False, prog_bar=True, logger=True)
       
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        """Validation step for the downstream classifier.

        Args:
            batch (torch.Tensor): Input batch of data.
            batch_idx (int): Index of the batch.

        Returns:
            dict: Dictionary containing the loss.
        """
        inputs = batch[:, :-1]
        labels = batch[:, -1]
        predictions = self(inputs)
        loss = self.criterion(predictions, labels)
        
        acc = torchmetrics.functional.accuracy(predictions, labels.int(), task="binary")
        prec = torchmetrics.functional.precision(predictions, labels, task="binary")
        recall = torchmetrics.functional.recall(predictions, labels.int(), task="binary")
        f1 = torchmetrics.functional.f1_score(predictions, labels.int(), task="binary")
        auroc = torchmetrics.AUROC(task="binary")
        auc = auroc(predictions, labels.int())
        apre = torchmetrics.AveragePrecision(task="binary")
        ap = apre(predictions, labels.int())
        
        logs = {
            "val/loss": loss, "val/accuracy": acc,
            "val/precision": prec, "val/recall": recall,
            "val/f1": f1, "val/auc": auc,
            "val/ap": ap
        }
        self.log_dict(logs, on_step=False, on_epoch=True, prog_bar=True, logger=True)
       
        return {"loss": loss}

    def test_step(self, batch, batch_idx):
        """Test step for the downstream classifier.

        Args:
            batch (torch.Tensor): Input batch of data.
            batch_idx (int): Index of the batch.

        Returns:
            dict: Dictionary containing the loss.
        """
        inputs = batch[:, :-1]
        labels = batch[:, -1]
        predictions = self(inputs)
        loss = self.criterion(predictions, labels)
        
        acc = torchmetrics.functional.accuracy(predictions, labels.int(), task="binary")
        prec = torchmetrics.functional.precision(predictions, labels, task="binary")
        recall = torchmetrics.functional.recall(predictions, labels.int(), task="binary")
        f1 = torchmetrics.functional.f1_score(predictions, labels.int(), task="binary")
        auroc = torchmetrics.AUROC(task="binary")
        auc = auroc(predictions, labels.int())
        apre = torchmetrics.AveragePrecision(task="binary")
        ap = apre(predictions, labels.int())
        
        logs = {
            "test/loss": loss, "test/accuracy": acc,
            "test/precision": prec, "test/recall": recall, 
            "test/f1": f1, "test/auc": auc,
            "test/ap": ap
        }
        self.log_dict(logs, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        return {"loss": loss}
    
    def configure_optimizers(self):
        """Configures the optimizers and learning rate scheduler.

        Returns:
            tuple: Optimizer and learning rate scheduler.
        """
        optimizer = torch.optim.Adam(params=self.parameters(), lr=self.cfg.train_params.downstream_lr, amsgrad=True)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.cfg.train_params.downstream_lr_step_size)
        return [optimizer], [scheduler]

  
