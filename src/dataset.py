import torch
from torch.utils.data import Dataset, DataLoader


class CMAPSSDataset(Dataset):
    """
    PyTorch Dataset wrapper for C-MAPSS sliding window sequences.
    """

    def __init__(self, X, y=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]


def get_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size=64):
    """
    Construct PyTorch DataLoaders for train, validation, and test sets.
    """
    train_dataset = CMAPSSDataset(X_train, y_train)
    val_dataset = CMAPSSDataset(X_val, y_val)
    test_dataset = CMAPSSDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
