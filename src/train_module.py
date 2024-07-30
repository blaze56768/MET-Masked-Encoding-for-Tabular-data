import os
import hydra
import sys
#sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#sys.path.insert(0, '../src')
from omegaconf import DictConfig
from datamodules.tabular_datamodule import TabularDataModule
from models.transformer_autoencoder import TransformerAutoEncoder
import lightning as L
from lightning.pytorch.loggers import WandbLogger
import wandb
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint


@hydra.main(version_base=None, config_path="../configs/pretrain", config_name="config.yaml")
def train(cfg: DictConfig) -> None:
    """Train function for pretraining a model using PyTorch Lightning.

    This function sets up the data loaders, initializes the model,
    and trains it using the configuration specified in `cfg`.

    Args:
        cfg (DictConfig): Configuration object containing all necessary parameters.
    """
    wandb_logger = WandbLogger(project=cfg.train_params.pretrain_wandb, group="toy", job_type="train")

    print("✅ Initializing the data loader...")
    #data_module = TelecomDataModule(cfg=cfg)
    data_module = TabularDataModule(cfg=cfg, name_labels = cfg.datamodule_params.name_labels, features_numerical=cfg.datamodule_params.features_numerical, features_categorical=cfg.datamodule_params.features_categorical)

    data_module.setup("fit")
    data_module.setup("validate")

    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    print("✅ Setup phase")
    model = TransformerAutoEncoder(cfg=cfg)

    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.pretrain_checkpoint,
        save_top_k=1,
        monitor="train/loss",
        mode="min",
        filename=cfg.datamodule_params.filename   # check how to access filename
    )
    
    trainer = L.Trainer(
        max_epochs=cfg.train_params.n_epochs,
        logger=wandb_logger,
        check_val_every_n_epoch=cfg.train_params.val_epochs,
        accelerator=cfg.train_params.device,
        callbacks=[checkpoint_callback, lr_monitor]
    )

    wandb_logger.watch(model, log="all")
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    checkpoint_callback.best_model_path

    wandb.finish()

if __name__ == "__main__":
    train()
