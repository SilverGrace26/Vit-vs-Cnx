from torch.utils.data import Dataset

class DatasetFromSubset(Dataset):
    def __init__(self, subset, transforms = None):
        super().__init__()
        self.subset = subset
        self.transforms = transforms

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transforms:
            x = self.transforms(x)
        return x, y

    def __len__(self):
        return len(self.subset)