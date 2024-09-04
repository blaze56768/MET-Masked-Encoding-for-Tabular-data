import hydra
from models import transformer_autoencoder
from omegaconf import DictConfig
from datamodules import TabularDataModule
from models import lr_classifier
import lightning as L
from lightning.pytorch.loggers import WandbLogger
import wandb
from utils import utils_funcs
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import LearningRateMonitor,ModelCheckpoint


@hydra.main(config_path = "../configs/pretrain",config_name ="config.yaml")  
def train(cfg : DictConfig)->None:
    wandb_logger = WandbLogger(project=cfg.downstream_wandb,group="toy",job_type = "train")
    
    print("✅Initializing the data loader...")
    sslloader = TabularDataModule(cfg=cfg)   
    

    sslloader.setup("fit")
    sslloader.setup("val")
    
    cfg.lr_step_size = int(sslloader.dataset_train.data.shape[0]/cfg.datamodule_params.batch_size)*cfg.datamodule_params.downstream_n_epochs
    
    train_loader =sslloader.train_dataloader()
    val_dataloader =sslloader.val_dataloader()
       
    print("✅Setup phase")  
    
    model = lr_classifier.LogisticRegressionClassifier(cfg=cfg)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(dirpath=cfg.downstream_checkpoint, save_top_k=1, monitor="train/ap", mode="max", filename=cfg.train_params.filename)
    trainer = L.Trainer(max_epochs=cfg.train_params.downstream_n_epochs, logger = wandb_logger, check_val_every_n_epoch=cfg.train_params.val_epochs, callbacks=[checkpoint_callback,lr_monitor])
    wandb_logger.watch(model,log="all")
    trainer.fit(model=model,train_dataloaders=train_loader,val_dataloaders=val_dataloader)
    checkpoint_callback.best_model_path
    
    return cfg

if __name__ == "__main__":
    train()
