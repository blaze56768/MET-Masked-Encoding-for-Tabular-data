__author__ = "Luis Felipe Villa Arenas"
__copyright__ = "Deutsche Telekom"

from pytorch_lightning.loggers.wandb import WandbLogger
import logging
import lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from omegaconf import DictConfig, OmegaConf
from typing import List
from torch import empty

def get_logger(name: str = __name__, level: int = logging.INFO) -> logging.Logger:
    """Initializes a multi-GPU-friendly Python logger.

    This logger ensures that logs are only output from the rank zero process 
    in a multi-GPU setup to avoid duplicate logs.

    Args:
        name (str): Name of the logger.
        level (int): Logging level.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Ensure all logging levels are marked with the rank zero decorator
    for lvl in ("debug", "info", "warning", "error", "exception", "fatal", "critical"):
        setattr(logger, lvl, rank_zero_only(getattr(logger, lvl)))

    return logger

@rank_zero_only
def log_hyperparameters(
    config: DictConfig,
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    trainer: pl.Trainer,
    callbacks: List[pl.Callback]
) -> None:
    """Logs hyperparameters and model information to all loggers.

    This method controls which parameters from the Hydra config are saved by Lightning loggers.
    Additionally, it logs the number of trainable and non-trainable model parameters.

    Args:
        config (DictConfig): Configuration object from Hydra.
        model (pl.LightningModule): PyTorch Lightning model.
        datamodule (pl.LightningDataModule): PyTorch Lightning data module.
        trainer (pl.Trainer): PyTorch Lightning trainer.
        callbacks (List[pl.Callback]): List of callbacks used in training.
    """
    hparams = {}

    # Choose which parts of the Hydra config will be saved to loggers
    hparams["trainer"] = config.get("trainer", {})
    hparams["model"] = config.get("model", {})
    hparams["datamodule"] = config.get("datamodule", {})
    if "callbacks" in config:
        hparams["callbacks"] = config["callbacks"]

    # Save number of model parameters
    hparams["model/params_total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params_trainable"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    hparams["model/params_not_trainable"] = sum(p.numel() for p in model.parameters() if not p.requires_grad)

    # Send hyperparameters to all loggers
    trainer.logger.log_hyperparams(hparams)

    # Disable logging any more hyperparameters for all loggers
    trainer.logger.log_hyperparams = empty