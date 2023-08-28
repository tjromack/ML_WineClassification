import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class WineQualityDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, index):
        #Convert data and target to PyTorch tensors
        data = torch.tensor(self.dataframe.iloc[index, :-1].values, dtype=torch.float32)
        target = torch.tensor(self.dataframe.iloc[index, -1], dtype=torch.long)
        return data, target

def get_dataloader(batch_size=32):
    preprocessed_data_path = "C:/Users/tjrom/Desktop/ml_wine/cleaned_winequality-red.csv"
    wine_data_processed = pd.read_csv(preprocessed_data_path)

    train_data, test_data = train_test_split(wine_data_processed, test_size=0.2, random_state=42)

    train_dataset = WineQualityDataset(train_data)
    test_dataset = WineQualityDataset(test_data)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader

# Testing loop to avoid running them every time you import from loader.py

# for batch_data, batch_labels in train_loader:
#     # Here goes your training code
#     pass

# for batch_data, batch_labels in test_loader:
#     # Here goes your evaluation code
#     pass
