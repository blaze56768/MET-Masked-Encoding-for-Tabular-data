from torch.utils.data import DataLoader
import lightning as L
from datamodules.datasets.tabular_dataset import TabularDataset


class TabularDataModule(L.LightningDataModule):
    """
    DataModule for Tabular data.

    This module prepares the training, validation, and test datasets
    for the tabular data and provides the corresponding DataLoaders.

    Args:
        cfg (object): Configuration object containing dataset paths and batch size.
    """
    
    def __init__(self, 
                 cfg,
                 name_labels,
                 features_numerical,
                 features_categorical):
        """
        Initializes the TabularDataModule with configuration parameters.
        
        Args:
            cfg (object): Configuration object containing dataset paths and batch size.
        """
        super().__init__()
        self.cfg = cfg

        # added
        self.name_labels = name_labels
        self.features_numerical = features_numerical
        self.features_categorical = features_categorical


    def setup(self, stage=None):
        """
        Sets up the datasets for different stages of training.

        Args:
            stage (str): The stage of setup ('fit', 'validate', 'test').
        """
        if stage == "fit":
            self.train_dataset = TabularDataset(self.cfg.datamodule_params.train_pth, self.name_labels, self.features_numerical, self.features_categorical) # added
        elif stage == "test" or stage == "validate":
            self.test_dataset = TabularDataset(self.cfg.datamodule_params.test_pth, self.name_labels, self.features_numerical, self.features_categorical) # added
        else:
            raise ValueError(f"Unknown stage: {stage}")

    def train_dataloader(self):
        """
        Returns the DataLoader for the training dataset.

        Returns:
            DataLoader: DataLoader for the training dataset.
        """
        return DataLoader(self.train_dataset, 
                          batch_size=self.cfg.datamodule_params.batch_size, 
                          shuffle=True,
                          num_workers=self.cfg.datamodule_params.num_workers)
    
    def val_dataloader(self):
        """
        Returns the DataLoader for the validation dataset.

        Returns:
            DataLoader: DataLoader for the validation dataset.
        """
        return DataLoader(self.test_dataset, 
                          batch_size=self.cfg.datamodule_params.batch_size, 
                          shuffle=False,
                          num_workers=self.cfg.datamodule_params.num_workers)

    def test_dataloader(self):
        """
        Returns the DataLoader for the test dataset.

        Returns:
            DataLoader: DataLoader for the test dataset.
        """
        return DataLoader(self.test_dataset, 
                          batch_size=self.cfg.datamodule_params.batch_size, 
                          shuffle=False,
                          num_workers=self.cfg.datamodule_params.num_workers)


