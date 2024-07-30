from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import torch

class TabularDataset(Dataset):
    """Custom Dataset for loading data from a CSV file.

    Args:
        csv_path (str): Path to the CSV file containing the data.
    """
    
    def __init__(self, 
                 csv_path,
                 name_labels,
                 features_numerical,
                 features_categorical):
        """
        Initializes the TabularDataset with the path to the CSV file.
        
        Args:
            csv_path (str): Path to the CSV file containing the data.
        """
        super().__init__()
        self.csv_path = csv_path
        self.name_labels = list(name_labels)
        self.features_numerical = list(features_numerical)
        self.features_categorical = list(features_categorical)
        self.data = pd.read_csv(csv_path)

    def __len__(self):
        """
        Returns the number of rows in the dataset.

        Returns:
            int: Number of rows in the dataset.
        """
        return len(self.data)

    def __getitem__(self, index):
        """
        Retrieves a row from the dataset as a tensor.

        Args:
            index (int): Index of the row to retrieve.

        Returns:
            torch.Tensor: The row data as a tensor.
        """
        item = self.data.loc[index].copy()

        # Labels
        labels = item[self.name_labels].values
        labels = torch.tensor(labels, dtype=torch.float)

        # Input
        input = item[self.features_numerical + self.features_categorical].values
        input = torch.tensor(input).type(torch.FloatTensor) 


        return input, labels
