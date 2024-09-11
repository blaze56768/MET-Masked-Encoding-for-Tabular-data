import hydra
from models import transformer_autoencoder
from omegaconf import DictConfig
from datamodules.tabular_datamodule import TabularDataModule
from models import lr_classifier
import lightning as L
from lightning.pytorch.loggers import WandbLogger
import wandb
from utils import utils_funcs
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import LearningRateMonitor,ModelCheckpoint

@hydra.main(version_base=None, config_path="../configs/pretrain", config_name="config.yaml")
def train(cfg : DictConfig)->None:
    wandb_logger = WandbLogger(project=cfg.train_params.downstream_wandb, group="toy", job_type = "train")
    
    print("✅Initializing the data loader...")
    data_module = TabularDataModule(cfg=cfg, name_labels = cfg.datamodule_params.name_labels, features_numerical=cfg.datamodule_params.features_numerical, features_categorical=cfg.datamodule_params.features_categorical)
    

    data_module.setup("fit")
    data_module.setup("validate")
    
    cfg.train_params.lr_step_size = int(data_module.train_dataset.data.shape[0]/cfg.datamodule_params.batch_size)*cfg.train_params.downstream_n_epochs
    
    train_loader = data_module.train_dataloader()
    val_dataloader = data_module.val_dataloader()
       
    print("✅Setup phase")  
    model = lr_classifier.LogisticRegressionClassifier(cfg=cfg)

    lr_monitor = LearningRateMonitor(logging_interval='step')

    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.downstream_checkpoint, 
        save_top_k=1, 
        monitor="train/ap", 
        mode="max", 
        filename=cfg.train_params.filename)
    

    trainer = L.Trainer(
        max_epochs=cfg.train_params.downstream_n_epochs, 
        logger = wandb_logger, 
        check_val_every_n_epoch=cfg.train_params.val_epochs, 
        callbacks=[checkpoint_callback,lr_monitor]
        )
    
    wandb_logger.watch(model,log="all")
    trainer.fit(model=model,train_dataloaders=train_loader,val_dataloaders=val_dataloader)
    checkpoint_callback.best_model_path
    
    return cfg

if __name__ == "__main__":
    train()
